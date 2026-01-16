import torch
import numpy as np
import wandb
from timeit import default_timer
from my_utils import parse_args, HsLoss_real, FVM_2D
from utils.mesh_utils import BatchedAngularMeshRavel
from utils.dataloader import Cylinder_data
import socket
import yaml
import imageio
import sys
args = parse_args()

from utilities import *
from models.geo_FNO import *
from visualizers.KF_flow import plot_evaluation_gif
from utils.gpu_utils import peripheral_setup
from utils.globalise import sweep_agent_wrapper, setup_output_dir
from dissipative_utils import sample_uniform_spherical_shell, linear_scale_dissipative_target
torch.manual_seed(42)
np.random.seed(42)

'''
TODO: 
- Geo-FNO concatenates the grid-coordinates (as does normal FNO). MNO does not do this.
- MNO models widths and modes are bigger than geo-FNO.
- The Ravler, at least while plotted, flips upside down. (ensure gradient is still calculated)
'''
# HPC or Local
if socket.gethostname() == 'DESKTOP-157DQSC':
    device = torch.device('cpu')
    data_path = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies v2"
    mesh_path = r"C:\Users\Noahc\Documents\USYD\tutorial"
    #sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Geo-FNO')
    sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
    DEBUG = True
else:
    device = torch.device('cuda')
    data_path = "/home/n.foster/datasets"
    mesh_path = "/home/n.foster/datasets"
    #sys.path.insert(0, r'/home/n.foster/Geo-FNO')
    sys.path.insert(0, r'/home/n.foster/Torch_VFM')
    DEBUG = False

#from airfoils.naca_geofno import FNO2d
from src.openfoam_utils.preload_mesh import preprocessed_OpenFOAM_mesh

class RegulatorSampler():
    def __init__(self, 
                 radii, 
                 shape, 
                 scale_down_factor = 0.5, 
                 weight = 0.01, 
                 sampling_fn = sample_uniform_spherical_shell,
                 target_fn = linear_scale_dissipative_target,
                 loss_fn=nn.MSELoss(reduction='mean')):
        
        self.sampling_fn = sampling_fn
        self.target_fn = target_fn
        self.loss_function = loss_fn
        self.weight = weight
        self.scale_down_factor = scale_down_factor

        # get shape((S, S, 1))
        self.shape = shape
        # get radii
        self.radii = radii

    def get_input(self, batch_s):
        return torch.tensor(self.sampling_fn(batch_s, self.radii, self.shape), dtype=torch.float)
    
    def get_target(self, x_diss):
        return self.target_fn(x_diss, self.scale_down_factor)
    
    def loss_fn(self, out, y):
        return self.weight * self.loss_function(out, y)

def train_batch(model, x, y, loss_fn, sampling_class=None):
    if len(x.shape) == 3: x = x.unsqueeze(-1)
    if len(y.shape) == 3: y = y.unsqueeze(-1)
    
    out = model(x)
    loss = loss_fn(out, y)
    loss_dict = {'hx_data_loss':loss.item()}

    if sampling_class is not None:
        x_diss = sampling_class.get_input(x.shape[0]).to(device)
        assert(x_diss.shape == x.shape)
        y_diss = sampling_class.get_target(x_diss).to(device)
        out_diss = model(x_diss).reshape(-1, y.shape[-1])
        diss_loss = sampling_class.loss_fn(out_diss, y_diss.reshape(-1, y.shape[-1]))
        loss_dict['diss_loss'] = diss_loss.item()

        loss += diss_loss

    return loss, loss_dict
        
def eval_batch(model, x, y, loss_fn):
    if len(x.shape) == 3: x = x.unsqueeze(-1)
    if len(y.shape) == 3: y = y.unsqueeze(-1)
    loss_dict = {}
    
    with torch.no_grad():
        out = model(x)
    loss_dict['testing/l2_data_loss'] = loss_fn(out, y, k=0).item()
    loss_dict['testing/h1_data_loss'] = loss_fn(out, y, k=1).item()
    loss_dict['testing/h2_data_loss'] = loss_fn(out, y, k=2).item()

    return loss_dict

def eval_longrollout(model, starting_u, T, S, H=None):
    if len(starting_u.shape) == 3:
        starting_u = starting_u.unsqueeze(-1)

    if H is None:
        H = S

    channel_n = starting_u.shape[-1]
    pred = torch.zeros_like(starting_u).repeat(T,1,1,1)
    out = starting_u.reshape(1,S,H,channel_n).to(device)
    pred[0,:,:,:] = out.view(S,H,channel_n)
    with torch.no_grad():
        for i in range(T-1):
            out = model(out)
            pred[i+1,:,:,:] = out.view(S,H,channel_n)
    
    return pred

def pipeline(config):

    grad_method     = config['parameters']['grad_method']
    epochs          = config['parameters']['epochs']
    learning_rate   = config['parameters']['learning_rate']
    scheduler_step  = config['parameters']['scheduler_step']
    scheduler_gamma = config['parameters']['scheduler_gamma']
    batch_size      = config['parameters']['batch_size'] if not DEBUG else 2
    in_dim          = config['dataset_params']['input_dim']
    out_dim         = config['dataset_params']['output_dim']
    k               = config['parameters']['soblev_norm_order']
    modes           = config['parameters']['modes']
    width           = config['parameters']['width']
    dataset_name    = config['dataset_params']['dataset_name']
    dataset_split   = config['dataset_params']['split']
    dataset_sub     = config['dataset_params']['sub']
    openfoam_case   = f"{mesh_path}/{config['dataset_params']['openfoam_case_dir']}"
    dataset_path    = f'{data_path}/{dataset_name}'

    dataset_radii   = config['dataset_params']['radii']
    dataset_angles  = config['dataset_params']['angles']

    assert config['dataset_params']['name'] == 'Cylinder' 
    assert config['parameters']['grad_method'] == 'FVM'

    # TODO: fix this dataloader
    coord_points = np.load(dataset_path[:-4]+'_coords.npy')
    Ravler = BatchedAngularMeshRavel(coord_points,m=dataset_radii,n=dataset_angles)

    train_loader, test_loader, W, H, max_norm = Cylinder_data(dataset_path, dataset_split, Ravler, batch_size=batch_size, sub=dataset_sub)
    
    # load in FVM mesh
    U_bc_dict = {'inlet':{ "type":'fixedValue', "value":[1,0,0]},
                'outlet':{ "type":'zeroGradient'},  
                'cylinder':{ "type":'noSlip'},
                'front':{ "type":'empty'},
                'back':{ "type":'empty'}
                }
    p_bc_dict = {'inlet':{ "type":'zeroGradient' },
                'outlet':{ "type":'zeroGradient' },
                'cylinder':{ "type":'zeroGradient'},
                'front':{ "type":'empty'},
                'back':{ "type":'empty'}
                }
    bc_dict = {'U':U_bc_dict, 'p':p_bc_dict}

    mesh = preprocessed_OpenFOAM_mesh(openfoam_case, dim=2, bc_dict=bc_dict)
    assert W*H == mesh.mesh.n_cells

    GRAD_FUNCTIONS = {'FVM': FVM_2D(mesh,Ravler,device=device)}
    
    if config['parameters']['dissipation']:
        regularizer = RegulatorSampler(radii=[max_norm, max_norm*4], shape=(W, H, out_dim), scale_down_factor=0.5, weight=0.01)
    else:
        regularizer = None
    
    #model = Net2d(in_dim=in_dim, out_dim=out_dim, domain_size=S, modes=20, width=64).to(device)
    model = FNO2d(modes1=modes*2, modes2=modes, width=width, input_dim=in_dim, output_dim=out_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    if grad_method == 'Spectral':
        train_loss_fn = HsLoss(group=True, size_average=False, k=k)
        eval_loss_fn = HsLoss(group=False, size_average=False)
    else:
        grad_calculator = GRAD_FUNCTIONS[grad_method]
        train_loss_fn = HsLoss_real(grad_calculator=grad_calculator, group=True, size_average=False, k=k)
        eval_loss_fn = HsLoss_real(grad_calculator=grad_calculator, group=False, size_average=False)
    
    for epoch in range(epochs):
        model.train()

        mini_batch_loss_dict = {}
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            loss, loss_dict = train_batch(model, x, y, train_loss_fn, sampling_class=regularizer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if mini_batch_loss_dict == {}: mini_batch_loss_dict = {k: [] for k in loss_dict}
            for key, value in loss_dict.items(): mini_batch_loss_dict[key].append(value)
            if DEBUG: break
        print(f'Progress: {epoch:3n}/{epochs:3n} - {i:3n}/{len(train_loader):3n} | Training Loss: {loss.item():3.4f}', end="\r", flush=True)

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            loss_dict = eval_batch(model, x, y, eval_loss_fn)
            if not all(key in mini_batch_loss_dict for key in loss_dict):
                for key in loss_dict: mini_batch_loss_dict[key] = []
            for key, value in loss_dict.items(): mini_batch_loss_dict[key].append(value)
            if DEBUG: break
        if DEBUG: break
        #print(f'Progress: {epoch:3n}/{epochs:3n} | Training Loss: {mini_batch_loss_dict["hx_data_loss"]:3.4f} | Validation Loss: {mini_batch_loss_dict["testing/l2_data_loss"]:3.4f}  ', end="\n", flush=True)

        # summarise epoch
        for key,value in mini_batch_loss_dict.items(): mini_batch_loss_dict[key] = np.mean(value)
        mini_batch_loss_dict['Epoch'] = epoch
        wandb.log(mini_batch_loss_dict)

        scheduler.step()
    
    # save model checkpoint
    setup_output_dir(config)
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, f"{config['peripheral']['out_dir']}/checkpoint_epoch_{epoch+1}.pth")

    # plot validation example
    #if not DEBUG:
    x, y = next(iter(test_loader))
    x = x.to(device)
    with torch.no_grad():
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        out = model(x)
    frames = plot_evaluation_gif(out,y, n=batch_size)
    imageio.mimsave(f"{config['peripheral']['out_dir']}/validation_sample.gif", frames, duration=0.5, loop=0)
    np.save(f"{config['peripheral']['out_dir']}/validation_sample.npy", out.cpu().numpy())
    media = wandb.Video(f"{config['peripheral']['out_dir']}/validation_sample.gif", caption=f"validation_sample")
    wandb.log({f"media/validation_sample": media})

    # perfrom long-rollout (at full res)
    test_u = Cylinder_data(dataset_path, dataset_split, Ravler, batch_size=batch_size, sub=dataset_sub, longrollout=True)
    
    #KF_flow_data(dataset_path, dataset_split, sub=1, T_in=dataset_T_in, T_out=dataset_T_out, longrollout=True,convert_to_u=velocity_f)
    out = eval_longrollout(model, test_u[[1],...], T=test_u.shape[0], S=W*dataset_sub, H=H*dataset_sub)
    
    cut_off = 5 if DEBUG else 100
    frames = plot_evaluation_gif(out[:cut_off,...],test_u[:cut_off,...], n=cut_off)
    imageio.mimsave(f"{config['peripheral']['out_dir']}/long_rollout_sample.gif", frames, duration=0.1, loop=0)
    np.save(f"{config['peripheral']['out_dir']}/long_rollout_sample.npy", out.cpu().numpy())
    media = wandb.Video(f"{config['peripheral']['out_dir']}/long_rollout_sample.gif", caption=f"long_rollout_sample")
    wandb.log({f"media/long_rollout_sample": media})


if __name__ == "__main__":
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = peripheral_setup(gpu_list=config['peripheral']['device'], seed=config['peripheral']['seed'])

    if args.sweep is not None:
        with open(args.sweep, 'r') as file:
            sweep_config = yaml.safe_load(file)
            sweep_id = wandb.sweep(sweep_config, project=config['wandb']['project'], entity=config['wandb']['entity'])
            config['peripheral']['out_dir'] = sweep_id
            wandb_sweep_controller = sweep_agent_wrapper(config)
            wandb_sweep_controller.assign_function(pipeline)
            wandb.agent(sweep_id, wandb_sweep_controller.run, count=config['wandb']['sweep_n'])
    else: 
        wandb.init(config=config,project=config['wandb']['project'], entity=config['wandb']['entity'], mode=args.wandb)
        pipeline(config)