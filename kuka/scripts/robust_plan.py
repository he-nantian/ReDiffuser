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

from rnd import RNDModel

parser = argparse.ArgumentParser()

parser.add_argument("--visible_gpu", type=str, default="6", help="visible GPU")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--plan_num", type=int, default=500, help="how many times to interface with env")
# parser.add_argument("--discount_power", type=float, default=10.0, help="how much we doubt at shorter horizons")
parser.add_argument("--GetMax", action="store_true", help="whether to get the max uncertainty")
parser.add_argument("--alpha_std", type=float, default=0.0, help="how much the variation of uncertainty guides the results")
# parser.add_argument("--beta_uncertainty", type=float, default=0.0, help="how much uncertainty guides the results")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
device = torch.device('cuda')


def soft_max(a):
    
    assert len(a.shape) == 1
    a -= np.max(a)
    
    return np.exp(a) / np.sum(np.exp(a))


def get_env_state(robot, cubes, attachments):
    joints = get_movable_joints(robot)
    joint_pos = get_joint_positions(robot, joints)

    for cube in cubes:
        pos, rot = get_pose(cube)
        pos, rot = np.array(pos), np.array(rot)

        if cube in attachments:
            attach = np.ones(1)
        else:
            attach = np.zeros(1)

        joint_pos = np.concatenate([joint_pos, pos, rot, attach], axis=0)

    return joint_pos


def execute(samples, env, idx=0):
    postprocess_samples = []
    robot = env.robot
    joints = get_movable_joints(robot)
    gains = np.ones(len(joints))

    cubes = env.cubes
    link = 8

    near = 0.001
    far = 4.0
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

    location = np.array([0.1, 0.1, 2.0])
    end = np.array([0.0, 0.0, 1.0])
    viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    attachments = set()

    states = [get_env_state(robot, cubes, attachments)]
    rewards = 0
    ims = []

    for sample in samples[1:]:
        p.setJointMotorControlArray(bodyIndex=robot, jointIndices=joints, controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7], positionGains=gains)

        attachments = set()
        # Add constraints of objects
        for j in range(4):
            contact = sample[14+j*8]

            if contact > 0.5:
                add_fixed_constraint(cubes[j], robot, link)
                attachments.add(cubes[j])
                env.attachments[j] = 1
            else:
                remove_fixed_constraint(cubes[j], robot, link)
                set_velocity(cubes[j], linear=[0, 0, 0], angular=[0, 0, 0, 0])
                env.attachments[j] = 0


        for i in range(10):
            p.stepSimulation()

        states.append(get_env_state(robot, cubes, attachments))

        _, _, im, _, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im)
        im = im.reshape((256, 256, 4))

        state = env.get_state()
        # print(state)
        reward = env.compute_reward()

        rewards = rewards + reward
        ims.append(im)
        # writer.append_data(im)

    attachments = {}
    env.attachments[:] = 0
    env.get_state()
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(robot, cubes, attachments)

    # writer.close()

    return state, states, ims, rewards


def choose_trajectory(n_samples, rnd_model_list, GetMax, alpha_std, H, beta_uncertainty, discount=1.0):
    
    model_num = len(rnd_model_list)
    sample_num, _,  horizon = n_samples.shape
    n_samples_pad = torch.concatenate(
        (
            n_samples, 
            torch.tile(n_samples[:, :, -1:], (1, 1, H-horizon))
        ), 
        dim=2
    )
    
    uncertainty_matrix = torch.zeros((model_num, sample_num)).cuda()
    for (i, rnd_model) in enumerate(rnd_model_list):
        uncertainty_matrix[i] = rnd_model(n_samples_pad)
    
    uncertainty_matrix = uncertainty_matrix / discount
    
    if GetMax:
        n_uncertainties = to_np(torch.max(uncertainty_matrix, dim=0)[0])
    else:
        n_uncertainties = to_np(
            torch.mean(uncertainty_matrix, dim=0) + alpha_std * torch.std(uncertainty_matrix, dim=0))
    
    n_probs = soft_max(-beta_uncertainty * n_uncertainties)
    
    sample_prob = np.random.random()
    cum_prob = 0.
    
    for (prob, samples) in zip(n_probs, n_samples):
        
        cum_prob += prob
        if sample_prob < cum_prob:
            break
    
    return torch.transpose(samples, 0, 1), np.mean(n_uncertainties)


def eval_episode(model, env, dataset, possible_horizon_list, rnd_model_list, 
                 discount_power, GetMax, alpha_std, sample_num, H, beta_uncertainty):
    
    initial_state = env.reset()

    obs_dim = dataset.obs_dim

    initial_state = (initial_state - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    initial_state = torch.Tensor(initial_state[None, None, None]).cuda()
    initial_state = (initial_state - 0.5) * 2

    conditions = [
           (0, obs_dim, initial_state),
    ]

    rewards = 0
    frames = []
    total_samples = []

    for i in range(3):
        
        best_samples = None
        lowest_uncertainty = np.inf
        best_horizon = 0
        
        for horizon in reversed(possible_horizon_list):
    
            n_samples = trainer.ema_model.conditional_sample(sample_num, conditions, horizon)
            n_samples = torch.clamp(n_samples, -1, 1)    # (sample_num, horizon, obs_dim)
            n_samples = torch.transpose(n_samples, 1, 2)    # (sample_num, obs_dim, horizon)
            
            samples, uncertainty_mean = choose_trajectory(n_samples, rnd_model_list, GetMax, alpha_std, H, 
                                                          beta_uncertainty, discount=(horizon/H)**discount_power)
            
            if uncertainty_mean < lowest_uncertainty:
                best_samples = samples.detach()
                best_horizon = horizon
                lowest_uncertainty = uncertainty_mean
        
        print("Best horizon has been chosen with {}".format(best_horizon))

        best_samples = (best_samples + 1) * 0.5
        best_samples = dataset.unnormalize(best_samples)
        best_samples = to_np(best_samples)
        
        initial_state, samples_list, frames_new, reward = execute(best_samples, env, idx=i)
        rewards = rewards + reward
        frames.extend(frames_new)
        total_samples.extend(samples_list)


        initial_state = (initial_state - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        initial_state = torch.Tensor(initial_state[None, None, None]).cuda()
        initial_state = (initial_state - 0.5) * 2

        conditions = [
               (0, obs_dim, initial_state),
        ]

    return rewards, total_samples, frames
    # return rewards


def to_np(x):
    return x.detach().cpu().numpy()

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])


#### hyperparameters

h = 96
H = 128
horizon_gap = 8
sample_num = 8
T = 100
dataset = KukaDataset(H)

env_name = f"kuka_{H}"

diffusion_path = f'logs/{env_name}/'
diffusion_epoch = 7

dataset = KukaDataset(H)

output_dim = 510
input_size = (39, H)

seed = args.seed
plan_num = args.plan_num
discount_power = 0.8
GetMax = args.GetMax
# alpha_std = 1.5
alpha_std = args.alpha_std
beta_uncertainty = 20.0

if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

savepath = f'test'
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
render_kwargs = {
    'trackbodyid': 2,
    'distance': 10,
    'lookat': [10, 2, 0.5],
    'elevation': 0
}

x = dataset[0][0].view(1, 1, H, obs_dim).cuda()
conditions = [
    (0, obs_dim, x[:, :, :1]),
]
trainer.ema_model.eval()

hidden_dims = [128, 128, 128]

config = Config(
    model_class=TimeConditionedMLP,
    time_dim=128,
    input_dim=obs_dim,
    hidden_dims=hidden_dims,
    output_dim=12,
    savepath="",
)

model = config.make()
model.to(device)

ckpt_path = "logs/kuka_cube_stack_classifier_new3/value_0.99/state_90.pt"
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)

counter = 0
map_tuple = {}
for i in range(4):
    for j in range(4):
        if i == j:
            continue

        map_tuple[(i, j)] = counter
        counter = counter + 1

rewards =  []
samples_full_list = []
frames_full_list = []

with open(os.path.join(savepath, "reward.txt"), "a") as log_f:
    log_f.write("\n{}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    log_f.write("seed: {}\t\tplan_num: {}\n".format(seed, plan_num))

rnd_model_list = []
for i in range(8):
    load_dir = "RNDmodel"
    rnd_model = RNDModel(input_size=input_size, output_dim=output_dim).cuda()
    rnd_model.target.load_state_dict(torch.load(os.path.join(load_dir, "target.pth")))
    rnd_model.predictor.load_state_dict(torch.load(os.path.join(load_dir, "predictor.pth")))
    rnd_model.eval()
    rnd_model_list.append(rnd_model)

possible_horizon_list = list(range(h, H+1, horizon_gap))

for i in range(plan_num):
    print("Episode: {} / {}".format(i+1, plan_num))
    reward, total_samples, frames = eval_episode(model, env, dataset, possible_horizon_list, rnd_model_list, 
                                                discount_power, GetMax, alpha_std, sample_num, H, beta_uncertainty)
    # reward = eval_episode(model, env, dataset, savepath, idx=i)
    rewards.append(reward)
    samples_full_list.extend(total_samples)
    frames_full_list.extend(frames)

rewards = np.array(rewards)
with open(os.path.join(savepath, "reward.txt"), "a") as log_f:
    print("rewards mean: {}\nrewards std: {}".format(np.mean(rewards / 3), np.std(rewards / 3)))
    log_f.write("rewards mean: {}\nrewards std: {}\n".format(np.mean(rewards / 3), np.std(rewards / 3)))

samples_full_list = np.array(samples_full_list)
np.save(os.path.join(savepath, "execution.npy"), samples_full_list)

writer = get_writer(os.path.join(savepath, "execution.mp4"))

for frame in frames_full_list:
    writer.append_data(frame)

writer.close()

