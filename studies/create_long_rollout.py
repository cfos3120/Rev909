import numpy as np
import torch
import socket
import sys
import os
import socket
import wandb
from pathlib import Path
import time
from functools import wraps
from contextlib import contextmanager


if socket.gethostname() == 'DESKTOP-157DQSC':
    sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
    sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Rev909')
    device = torch.device('cpu')
    data_path = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies"#\2D_NS_Re40.npy"
    DEBUG = True
else:
    sys.path.insert(0, r'/home/n.foster/Torch_VFM')
    sys.path.insert(0, r'/home/n.foster/Rev909')
    device = torch.device('cuda')
    data_path = "/home/n.foster/datasets" #2D_NS_Re500.npy"
    DEBUG = False

from models.fno_2d import *
from main import eval_longrollout
from utils.dataloader import KF_flow_data

@contextmanager
def suppress_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def retrieve_wandb_config(target_run_name, project, entity="cfos3120-acfr-usyd"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    # Search for the run by name
    target_run = next((run for run in runs if run.name == target_run_name), None)
    if target_run:
        return target_run.config, target_run.id
    else:
        for run in runs:
            print(run.name, run.id)
        raise KeyError(f"Run with name '{target_run_name}' not found in project '{project}'.")

def eval_model_long_rollout(config, file_name):
    in_dim          = config['dataset_params']['input_dim']
    out_dim         = config['dataset_params']['output_dim']
    dataset_name    = config['dataset_params']['dataset_name']
    dataset_split   = config['dataset_params']['split']
    dataset_sub     = config['dataset_params']['sub']
    dataset_T_in    = config['dataset_params']['T_in']
    dataset_T_out   = config['dataset_params']['T_out']
    dataset_path = f'{data_path}/{dataset_name}'
    if config['dataset_params']['name'] == 'Kolmogorov Flow' and out_dim == 2:
        velocity_f = True
    else:
        velocity_f = False

    #
    T_rollout = 10000
    with suppress_print():
        test_u = KF_flow_data(dataset_path, dataset_split, sub=1, T_in=dataset_T_in, T_out=dataset_T_out, longrollout=True,convert_to_u=velocity_f)
    S = test_u.shape[1]
    model = Net2d(in_dim=in_dim, out_dim=out_dim, domain_size=S, modes=20, width=64).to(device)

    ckpt_path = f"{config['peripheral']['out_dir']}/checkpoint_epoch_50.pth"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    start_t = time.perf_counter()
    out = eval_longrollout(model, test_u[[1],...], T=T_rollout, S=S*dataset_sub)
    end_t = time.perf_counter()
    print(f'Completed in: {(end_t-start_t)/60:4f} minutes')
    np.save(f"{config['peripheral']['out_dir']}/{file_name}.npy", out.cpu().numpy())

if __name__ == "__main__":

    cwd = Path.cwd()        # current working directory
    parent = cwd.parent     # parent directory
    path = rf'{Path(__file__).parent.parent}/results/Rev909/SWEEP-cgaf3jfu'

    project = 'Rev909'
    for folder in os.listdir(path):
        print(f'Rolling out case: {folder}', end=' ')
        config_file, run_id = retrieve_wandb_config(folder.rsplit("-", 1)[0], project, entity="cfos3120-acfr-usyd")
          
        # need to correct for some reason on HPC
        if socket.gethostname() != 'DESKTOP-157DQSC':
            config_file['peripheral']['out_dir'] = f'{path}/{folder}'
        
        eval_model_long_rollout(config=config_file, file_name='long_rollout_10000.npy')

    print('Rollout of sweep complete')

