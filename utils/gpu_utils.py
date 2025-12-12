import torch
import numpy as np
import random
import os

def peripheral_setup(gpu_list, seed):
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    assert isinstance(gpu_list, list), ValueError('gpu_list should be a list.')

    if torch.cuda.is_available():
        device = 'cuda' if gpu_list else 'cpu'
    else:
        device = 'cpu'
    # if torch.cuda.is_available() and device == 'cuda':
    if device == 'cuda':
        gpu_list = ','.join(str(x) for x in gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        #torch.backends.cudnn.benchmark = True  # type:ignore
        #torch.backends.cudnn.deterministic = True  # type:ignore
        #torch.multiprocessing.set_sharing_strategy('file_system')
        #torch.autograd.set_detect_anomaly(True)
        print("Using CUDA...")
        print("GPU number: {}".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("Using CPU, GPU not available..")
    
    device = torch.device(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    return device