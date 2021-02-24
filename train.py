import argparse
import collections
import torch
import data_loader.data_loaders as module_data

import model.model as module_arch
import pytorch_lightning as pl
from parse_config import ConfigParser
from utils import prepare_device
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    logger = config.get_logger('train')
    logger.info(config.log_dir)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, config=config)
    logger.info(model)

    early_stop_mode, early_stop_monitor = config['trainer']['monitor']. split(' ')
    early_stop_callback = EarlyStopping(
        monitor=early_stop_monitor,
        min_delta=0.00,
        patience=config['trainer']['early_stop'],
        verbose=False,
        mode=early_stop_mode
    )
    logger.info(f'Resume from file: {config.resume}')
    trainer = pl.Trainer(gpus=config['n_gpu'],
                         logger=tb_logger,
                         callbacks=[early_stop_callback],
                         limit_train_batches=config['trainer']['train_batches'],
                         limit_val_batches=config['trainer']['val_batches'],
                         limit_test_batches=config['trainer']['test_batches'],
                         default_root_dir=config['trainer']['save_dir'],
                         resume_from_checkpoint=config.resume)
    trainer.fit(model, data_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
