import torch
import numpy as np
import sys
import socket
import yaml
import wandb
import time
from types import SimpleNamespace
import pickle

# HPC or Local
if socket.gethostname() == 'DESKTOP-157DQSC':
    device = torch.device('cpu')
    home_path =  "C:/Users/Noahc/Documents/USYD"
    data_path = f"{home_path}/PHD/0 - Work Space/Markov Studies v2"
    mesh_path = f"{home_path}/tutorial"
    git_path = f'{home_path}/PHD/8 - Github'
    ckpt_path = f'{home_path}/PHD/0 - Work Space/Markov Studies v2/Cylinder/exports'
    DEBUG=True
else:
    device = torch.device('cuda')
    home_path =  "/home/n.foster"
    data_path = f"{home_path}/datasets"
    mesh_path = f"{home_path}/openfoam_Cases"
    git_path = home_path
    ckpt_path = f"{home_path}/Rev909/results/Cylinder"
    DEBUG=False

sys.path.insert(0, f'{git_path}/Rev909')
from utils.mesh_utils import BatchedAngularMeshRavel
from utils.loss_functions import RegulatorSampler
from create_long_rollout import retrieve_wandb_config
from models.geo_FNO import FNO2d
from utils.dataloader import Cylinder_data
from my_utils import HsLoss_real, FVM_2D

sys.path.insert(0, rf'{git_path}/Torch_VFM')
from src.openfoam_utils.preload_mesh import preprocessed_OpenFOAM_mesh

sys.path.insert(0, rf'{git_path}/loss-landscape')
import net_plotter
import copy
from plot_surface import setup_surface_file, name_surface_file
import projection as proj
import scheduler

def get_losses(model, dataset_batch, sampling_class, loss_fn):
    x, y = [i.to(device) for i in dataset_batch]
    if len(x.shape) == 3: x = x.unsqueeze(-1)
    if len(y.shape) == 3: y = y.unsqueeze(-1)
    loss_dict = {}
    with torch.no_grad():
        out = model(x)
        loss_dict['testing/l2_data_loss'] = loss_fn(out, y, k=0).item()
        loss_dict['testing/h1_data_loss'] = loss_fn(out, y, k=1).item()
        loss_dict['testing/h2_data_loss'] = loss_fn(out, y, k=2).item()

        x_diss = sampling_class.get_input(x.shape[0]).to(device)
        assert(x_diss.shape == x.shape)
        y_diss = sampling_class.get_target(x_diss).to(device)
        out_diss = model(x_diss).reshape(-1, y.shape[-1])
        diss_loss = sampling_class.loss_fn(out_diss, y_diss.reshape(-1, y.shape[-1]))
        loss_dict['diss_loss'] = diss_loss.item()
    
    return loss_dict


def fno_crunch_state(results_dict, model, loss_fn, sampling_class, dataloader, s, d):

    xcoordinates = results_dict['xcoordinates']
    ycoordinates = results_dict['ycoordinates']
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    coords = np.c_[xcoord_mesh.ravel(),ycoord_mesh.ravel()]

    for i, coord_idx in enumerate(range(coords.shape[0])):
        coord = coords[coord_idx]
        print(f'Computing for {i+1}/{coords.shape[0]} coordinates with x={coord[0]:.2f}, y={coord[1]:.2f}', end= ' ')
        t0 = time.perf_counter()

        net_plotter.set_states(model, s, d, coord)

        mini_batch_loss_dict = {}
        for j, dataset_batch in enumerate(dataloader):
            loss_dict = get_losses(model, dataset_batch, sampling_class, loss_fn)
            if not all(key in mini_batch_loss_dict for key in loss_dict):
                for key in loss_dict: mini_batch_loss_dict[key] = []
            for key, value in loss_dict.items(): mini_batch_loss_dict[key].append(value)
            if DEBUG and j > 0: break
        # summarize over all batches
        for key,value in mini_batch_loss_dict.items(): mini_batch_loss_dict[key] = np.mean(value)

        results_dict['L2_loss'].append(mini_batch_loss_dict['testing/l2_data_loss']) 
        results_dict['H1_loss'].append(mini_batch_loss_dict['testing/h1_data_loss']) 
        results_dict['H2_loss'].append(mini_batch_loss_dict['testing/h2_data_loss'])
        results_dict['Reg_loss'].append(mini_batch_loss_dict['diss_loss'])
        
        t1 = time.perf_counter()
        print(f'completed with time: {((t1 - t0) / 60):.2f}s')
        if DEBUG and i > 0: break

    return results_dict

if __name__ == '__main__':

    comm, rank, nproc = None, 0, 1 # Hard coded

    # Load in configurations
    config_file = f'{git_path}/Rev909/configs/loss_surface.yaml'
    with open(config_file, 'r') as file:
        args = yaml.safe_load(file)
        args = SimpleNamespace(**args)

    # Set multi-GPU support
    if device == torch.device('cuda'):
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
                (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))
    
    # Some corrections
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
    args.xnum = int(args.xnum)
    args.ynum = int(args.ynum)

    # Load in configs:
    config, run_id1 = retrieve_wandb_config(target_run_name=args.model1, project=args.wandb_project, entity=args.wandb_entity)
    if args.model2 is not None:
        __, run_id2 = retrieve_wandb_config(target_run_name=args.model2, project=args.wandb_project, entity=args.wandb_entity)
    if args.model3 is not None:
        __, run_id3 = retrieve_wandb_config(target_run_name=args.model3, project=args.wandb_project, entity=args.wandb_entity)

    args_model = config['parameters']
    args_data  = config['dataset_params']
    dataset_path    = f"{data_path}/{args_data['dataset_name']}"
    openfoam_case   = f"{mesh_path}/{args_data['openfoam_case_dir']}"
    dataset_radii   = args_data['radii']
    dataset_angles  = args_data['angles']
    
    # Load in dataset and ravler
    coord_points = np.load(dataset_path[:-4]+'_coords.npy')
    Ravler = BatchedAngularMeshRavel(coord_points,m=dataset_radii,n=dataset_angles, device=device)
    assert args_model['input_xy'] == True
    coord_points = Ravler.to(torch.tensor(coord_points,dtype=torch.float32), forward=True).to(device)

    train_loader,__, W, H, max_norm = Cylinder_data(dataset_path, args_data['split'], Ravler, batch_size=args_model['batch_size'], sub=args_data['sub'], longrollout=False)
    assert W == dataset_radii and H == dataset_angles

    # load in FVM mesh
    U_bc_dict = {'inlet':{ "type":'fixedValue', "value":[1,0,0]},
                'outlet':{ "type":'zeroGradient'},  
                'cylinder':{ "type":'noSlip'},
                'front':{ "type":'empty'},
                'back':{ "type":'empty'}
                }
    p_bc_dict = {'inlet':{ "type":'zeroGradient'},
                'outlet':{ "type":'zeroGradient'},
                'cylinder':{ "type":'zeroGradient'},
                'front':{ "type":'empty'},
                'back':{ "type":'empty'}
                }
    bc_dict = {'U':U_bc_dict, 'p':p_bc_dict}

    mesh = preprocessed_OpenFOAM_mesh(openfoam_case, dim=2, bc_dict=bc_dict, device=device)
    assert W*H == mesh.mesh.n_cells

    # Load in Model
    model = FNO2d(modes1=args_model['modes']*2, 
                  modes2=args_model['modes'], 
                  width=args_model['width'], 
                  input_dim=args_data['input_dim'], 
                  output_dim=args_data['output_dim'], 
                  grid=coord_points).to(device)
    
    ckpt_path1 = f"{ckpt_path}/{args.wandb_sweep}/{args.model1}-{run_id1}/{args.ckpt_name}"

    ckpt1 = torch.load(ckpt_path1, map_location=device)
    model.load_state_dict(ckpt1['model_state_dict'])

    # Loss-surface Preperation:
    w = net_plotter.get_weights(model) # initial parameters
    s = copy.deepcopy(model.state_dict()) # deepcopy since state_dict are references

    # create own storage dictionary and directions:
    results_dict = {'L2_loss':[], 
                    'H1_loss':[], 
                    'H2_loss':[], 
                    'Reg_loss':[],
                    'xcoordinates': np.linspace(args.xmin, args.xmax, num=args.xnum),
                    'ycoordinates': np.linspace(args.ymin, args.ymax, num=args.ynum)
                    }

    if args.model2:
        model2 = model.copy()
        ckpt_path2 = f"{ckpt_path}/{args.wandb_sweep}/{args.model2}-{run_id2}/{args.ckpt_name}"
        ckpt2 = torch.load(ckpt_path2, map_location=device)
        model2.load_state_dict(ckpt2['model_state_dict'])
        xdirection = net_plotter.create_target_direction(model, model2, args.dir_type)
    else:
        xdirection = net_plotter.create_random_direction(model, args.dir_type, args.xignore, args.xnorm)
    
    if args.y:
        if args.same_dir:
            ydirection = xdirection
        else:
            raise NotImplementedError
        
    directions = [xdirection, ydirection]

    # Loss functions
    sampling_class = RegulatorSampler(radii=[max_norm, max_norm*4], shape=(W, H, args_data['output_dim']), scale_down_factor=0.5, weight=0.01)
    grad_calculator=FVM_2D(mesh,Ravler,volume_weighting=False,device=device)
    eval_loss_fn = HsLoss_real(grad_calculator=grad_calculator, group=False, size_average=False)

    results_dict= fno_crunch_state(results_dict, 
                                   model, 
                                   loss_fn=eval_loss_fn, 
                                   sampling_class=sampling_class, 
                                   dataloader=train_loader, 
                                   s=s, 
                                   d=directions)

    save_path = f"{ckpt_path}/{args.wandb_sweep}/{args.model1}-{run_id1}/loss_surface.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

    if DEBUG:
        with open(save_path, "rb") as f:
            my_dict = pickle.load(f)
            print(results_dict)














        