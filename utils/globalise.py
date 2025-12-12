# moves config items to Global Variables
import yaml
import wandb
import os
import pprint
import time

def setup_output_dir(config):
    
    # set timestamp
    config['peripheral']['time_stamp'] = time.strftime('%Y%m%d%H%M%S')
    
    # extra sweep directory
    if wandb.run.sweep_id is None:
        sweep_str=''
    else: 
        sweep_str=f'SWEEP-{wandb.run.sweep_id}'

    # folder name
    if wandb.run.name == '':
        prefix = 'RUN'
    elif wandb.run.name is None:
        prefix = 'dummy'
    elif wandb.run.name[:5] == 'dummy':
        prefix = 'dummy'
    else:
        prefix = f'{wandb.run.name}'

    # set dir
    out_dir = os.path.realpath(__file__).split('/')
    out_dir = os.path.join('/'.join(out_dir[:-2]), 'results/{}'.format(config['wandb']['project']), sweep_str, f'{prefix}-{wandb.run.id}')
    config['peripheral']['out_dir'] = out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def wandb_logger(metrics_list):
    wandb.define_metric("Epoch")
    wandb.define_metric('Learning Rate', step_metric="Epoch")
    wandb.define_metric('Training Loss', step_metric="Epoch")
    wandb.define_metric('Validation Loss', step_metric="Epoch")
    for metric_name in metrics_list:
        wandb.define_metric(f'Validation ({metric_name})', step_metric="Epoch")

class sweep_agent_wrapper():
    def __init__(self, base_config):
        self.base_config = base_config

    def assign_function(self, pipeline_function):
        self.pipeline = pipeline_function

    def run(self):
        if 'group' in self.base_config['wandb'].keys():
            group = self.base_config['wandb']['group']
        else: 
            group = None

        with wandb.init(config=None, group=group):
            self.base_config['parameters'].update(wandb.config)
            #self.new_config = wandb.config.update(self.base_config)

            self.pipeline(self.base_config)