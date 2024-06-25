import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.lbunet import LBUNet

from engine import *
import os

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):
    config.work_dir = '/data1/xujiahao/Project/LB-UNet/'
    log_dir = os.getcwd()
    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)
    
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'lbunet':
        model = LBUNet(num_classes=model_cfg['num_classes'], 
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()
    
    input_path = ''

    if os.path.exists(input_path):
        print('#----------Testing----------#')
        best_weight = torch.load(input_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
                val_loader,
                model,
                config.criterion,
                logger,
                config,
                path = 'ultimate'
            )


if __name__ == '__main__':
    config = setting_config
    main(config)