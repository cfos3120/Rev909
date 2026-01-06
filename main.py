import torch
import numpy as np
import wandb
from timeit import default_timer
from my_utils import parse_args, HsLoss_real, periodic_FVM, periodic_derivatives
from utils.dataloader import KF_flow_data
import socket
import yaml
import imageio
args = parse_args()

from utilities import *
from models.fno_2d import *
from visualizers.KF_flow import plot_evaluation_gif
from utils.gpu_utils import peripheral_setup
from utils.globalise import sweep_agent_wrapper, setup_output_dir
from dissipative_utils import sample_uniform_spherical_shell, linear_scale_dissipative_target
torch.manual_seed(42)
np.random.seed(42)

# HPC or Local
if socket.gethostname() == 'DESKTOP-157DQSC':
    device = torch.device('cpu')
    data_path = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies"#\2D_NS_Re40.npy"
else:
    device = torch.device('cuda')
    data_path = "/home/n.foster/datasets" #2D_NS_Re500.npy"


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

def eval_longrollout(model, starting_u, T, S):
    if len(starting_u.shape) == 3:
        starting_u = starting_u.unsqueeze(-1)

    channel_n = starting_u.shape[-1]
    pred = torch.zeros_like(starting_u).repeat(T,1,1,1)
    out = starting_u.reshape(1,S,S,channel_n).to(device)
    pred[0,:,:,:] = out.view(S,S,channel_n)
    with torch.no_grad():
        for i in range(T-1):
            out = model(out)
            pred[i+1,:,:,:] = out.view(S,S,channel_n)
    
    return pred

def pipeline(config):

    grad_method     = config['parameters']['grad_method']
    epochs          = config['parameters']['epochs']
    learning_rate   = config['parameters']['learning_rate']
    scheduler_step  = config['parameters']['scheduler_step']
    scheduler_gamma = config['parameters']['scheduler_gamma']
    batch_size      = config['parameters']['batch_size']
    out_dim         = config['dataset_params']['input_dim']
    in_dim          = config['dataset_params']['output_dim']
    k               = config['parameters']['soblev_norm_order']
    dataset_name    = config['dataset_params']['dataset_name']
    dataset_split   = config['dataset_params']['split']
    dataset_sub     = config['dataset_params']['sub']
    dataset_T_in    = config['dataset_params']['T_in']
    dataset_T_out   = config['dataset_params']['T_out']
    dataset_path = f'{data_path}/{dataset_name}'

    train_loader, test_loader, S, max_norm = KF_flow_data(dataset_path, dataset_split, batch_size=batch_size, sub=dataset_sub, T_in=dataset_T_in, T_out=dataset_T_out)
    
    GRAD_FUNCTIONS = {'FDM': periodic_derivatives,
                      'FVM': periodic_FVM(S=S,device=device),
                      #'Autograd':,
                      #'Spectral:'
                      }
    
    if config['parameters']['dissipation']:
        regularizer = RegulatorSampler(radii=[max_norm, max_norm*4], shape=(S, S, 1), scale_down_factor=0.5, weight=0.01)
    else:
        regularizer = None
    
    model = Net2d(in_dim=in_dim, out_dim=out_dim, domain_size=S, modes=20, width=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    grad_calculator = GRAD_FUNCTIONS[grad_method]

    if grad_method == 'Spectral':
        train_loss_fn = HsLoss(group=True, size_average=False, k=k)
        eval_loss_fn = HsLoss(group=False, size_average=False)
    else:
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
        print(f'Progress: {epoch:3n}/{epochs:3n} - {i:3n}/{len(train_loader):3n} | Training Loss: {loss.item():3.4f}', end="\r", flush=True)

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            loss_dict = eval_batch(model, x, y, eval_loss_fn)
            if not all(key in mini_batch_loss_dict for key in loss_dict):
                for key in loss_dict: mini_batch_loss_dict[key] = []
            for key, value in loss_dict.items(): mini_batch_loss_dict[key].append(value)
        
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
    test_u = KF_flow_data(dataset_path, dataset_split, sub=1, T_in=dataset_T_in, T_out=dataset_T_out, longrollout=True)
    out = eval_longrollout(model, test_u[[1],...], T=test_u.shape[0], S=S*dataset_sub)
    
    cut_off = 100
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