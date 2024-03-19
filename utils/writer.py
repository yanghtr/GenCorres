import os
import time
import torch

from pyutils import *
from loguru import logger
from tensorboardX import SummaryWriter


def get_state_info_str(state_info):
    state_info_str = f"epoch={state_info['epoch']:05d} it={state_info['b']:04d} " 
    for k, v in state_info.items():
        if 'loss' in k:
            state_info_str = f"{state_info_str}{k}={v:.6f} "
    for k, v in state_info.items():
        if 'err' in k:
            state_info_str = f"{state_info_str}{k}={v:.6f} "
    state_info_str = f"{state_info_str}|lr= "
    for lr in state_info['lr']:
        state_info_str = f"{state_info_str}{lr:.6f} "
    return state_info_str


class Writer():
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        if config.mode == 'train':
            self.summary_writer = SummaryWriter(get_directory(f"{log_dir}/summary/{config.mode}/{config.rep}"))
        elif config.mode == 'test_opt':
            self.summary_writer = SummaryWriter(get_directory(f"{log_dir}/summary/{config.mode}/{config.rep}/{config.epoch_continue}"))
        else:
            self.summary_writer = None


    def log_state_info(self, state_info):
        state_info_str = get_state_info_str(state_info)
        logger.info(state_info_str)


    def log_summary(self, state_info, global_step, mode):
        for k, v in state_info.items():
            if 'loss' in k:
                self.summary_writer.add_scalar(f'{mode}/{k}', v, global_step)
        for i, lr in enumerate(state_info['lr']):
            self.summary_writer.add_scalar(f'{mode}/lr{i}', lr, global_step)


    def save_checkpoint(self, ckpt_path, epoch, model, latent_vecs, optimizer):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_latent_vecs': latent_vecs.state_dict() if latent_vecs is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            },
            ckpt_path
        )
        logger.info(ckpt_path)


    def load_checkpoint(self, ckpt_path, model=None, latent_vecs=None, optimizer=None):
        # in-place load
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if model is not None:
            logger.info(f"load model from ${ckpt_path}")
            model.load_state_dict(ckpt["model_state_dict"])
        if latent_vecs is not None:
            logger.info(f"load lat_vecs from ${ckpt_path}")
            latent_vecs.load_state_dict(ckpt["train_latent_vecs"])
        if optimizer is not None:
            logger.info(f"load optimizer from ${ckpt_path}")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("loaded!")
        return ckpt["epoch"]

