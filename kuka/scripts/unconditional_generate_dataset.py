import os
import numpy as np
import torch
import pdb
import pybullet as p
import os.path as osp

import gym
import d4rl

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer_old import MixerUnet
from denoising_diffusion_pytorch.mixer import MixerUnet as MixerUnetNew
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
import environments
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

from denoising_diffusion_pytorch.utils.pybullet_utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity

from gym_stacking.env import StackEnv
from tqdm import tqdm

import time
import argparse

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
data_size = 3 * (128 ** 2)
batch_size = 8
H = 128
T = 100

seed = 0
if seed:
    print("Random seed: {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_dataset():

    env_name = f"kuka_{H}"

    diffusion_path = f'logs/{env_name}/'
    diffusion_epoch = 7

    dataset = KukaDataset(H)

    savepath = f'dataset_{H}_{T}'
    utils.mkdir(savepath)

    ## dimensions
    obs_dim = dataset.obs_dim

    model = TemporalUnet(
        horizon = H,
        transition_dim = obs_dim,
        cond_dim = H,
        dim = 128,
        dim_mults = (1, 2, 4, 8),
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        channels = 2,
        image_size = (H, obs_dim),
        timesteps = T,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).cuda()

    env = StackEnv(conditional=False)

    trainer = Trainer(
        diffusion,
        dataset,
        env,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 700001,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        fp16 = False,                     # turn on mixed precision training with apex
        results_folder = diffusion_path,
    )

    print(f'Loading: {diffusion_epoch}')
    trainer.load(diffusion_epoch)

    trainer.ema_model.eval()
    
    n_iterations = int(data_size / 3 / batch_size)
    print("Generating iterations: {}".format(n_iterations))
    
    decision_dataset = np.empty(shape=(0, obs_dim, H))

    for i in range(n_iterations):
        
        initial_state = torch.Tensor(env.reset())
        initial_state = (initial_state - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        initial_state = initial_state[None, None, None].cuda()
        initial_state = (initial_state - 0.5) * 2
        
        for j in range(3):
            
            conditions = [(0, obs_dim, initial_state)]
            samples = torch.clamp(trainer.ema_model.conditional_sample(batch_size, conditions), -1, 1)
            
            initial_state = samples[0][-1][None, None, None]
            
            samples = np.transpose(samples.detach().cpu().numpy(), (0, 2, 1))
            decision_dataset = np.concatenate((decision_dataset, samples))
        
        if (i+1) % 16 == 0:
            print("{}%  ----  Generate dataset of size:  {}/{}".format(
                int(100*(i+1)*batch_size*3/data_size), (i+1)*batch_size*3, data_size))
    
    np.random.shuffle(decision_dataset)
    np.save(os.path.join(savepath, "unconditional.npy"), decision_dataset)
    print("Decision dataset has been shuffled and saved with shape of {}".format(decision_dataset.shape))


if __name__ == "__main__":
    generate_dataset()