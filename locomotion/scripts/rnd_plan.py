import numpy as np
import torch

from tqdm import tqdm
import argparse
import time

from rnd_trajectory import RNDModel_halfcheetah, RNDModel_hopper, RNDModel_walker2d

import diffuser.sampling as sampling
import diffuser.utils as utils

import os
import time
# from joblib import Parallel, delayed


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'
    visible_gpu: str = '0'
    seed: int = 8
    num_eval: int = 10
    # lamda: float = 1 / 8
    scaling: float = 0.2
    # alpha: float = 20.0
    top_value_size: int = 2
    output_dim: int = 200

args = Parser().parse_args('plan')

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu


# Hyperparameters choosing for top_value_size and scaling
# Hyperparameters = {
#     "halfcheetah-medium-expert-v2": [4, 0.2], 
#     "halfcheetah-medium-v2": [2, 0.2], 
#     "halfcheetah-medium-replay-v2": [8, 0.2], 
    
#     "hopper-medium-expert-v2": [4, 0.2], 
#     "hopper-medium-v2": [2, 0.4], 
#     "hopper-medium-replay-v2": [2, 0.8], 
    
#     "walker2d-medium-expert-v2": [8, 0.2], 
#     "walker2d-medium-v2": [4, 0.2], 
#     "walker2d-medium-replay-v2": [4, 0.4]
# }


def soft_max(a):
    
    assert len(a.shape) == 1
    a -= np.max(a)
    
    return np.exp(a) / np.sum(np.exp(a))


def uncertainty_value_guide(RNDmodel, samples, scaling, is_halfcheetah, top_value_size):
    
    observations = samples.observations
    actions = samples.actions
    values = utils.to_np(samples.values)
    
    # top_value_size = int(lamda * observations.shape[0])
    # top_value_size = observations.shape[0]
    # top_value_size = 4
    observations = observations[:top_value_size]
    actions = actions[:top_value_size]
    values = values[:top_value_size]
    
    if is_halfcheetah:
        obs4model = observations.reshape(observations.shape[0], -1)
    else:
        obs4model = np.transpose(observations, axes=(0, 2, 1))
    
    uncertainties = RNDmodel(torch.Tensor(obs4model).cuda()).detach().cpu().numpy()
    
    # Z-score标准化
    # values = (values - np.mean(values)) / np.std(values)
    # uncertainties = (uncertainties - np.mean(uncertainties)) / np.std(uncertainties)
    # min-max归一化
    # values = (values - np.min(values)) / (np.max(values) - np.min(values))
    # uncertainties = (uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))

    # print(uncertainties)
    # print(utils.to_np(samples.values))
    # time.sleep(2)
    
    probs = soft_max(-scaling * uncertainties)
    
    sample_prob = np.random.random()
    cum_prob = 0.
    
    for (prob, action) in zip(probs, actions):
        
        cum_prob += prob
        if sample_prob < cum_prob:
            break
    
    return action[0]


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

output_dim = args.output_dim

if "halfcheetah" in args.dataset:
    input_size = 68
    # output_dim = 30
    RNDmodel = RNDModel_halfcheetah(input_size, output_dim).cuda()

elif "hopper" in args.dataset:
    input_size = (11, 32)
    # output_dim = 150
    RNDmodel = RNDModel_hopper(input_size, output_dim).cuda()

elif "walker2d" in args.dataset:
    input_size = (17, 32)
    # output_dim = 250
    RNDmodel = RNDModel_walker2d(input_size, output_dim).cuda()

else:
    raise NameError

RNDmodel_load_dir = f"RNDmodel/{args.dataset}"

RNDmodel.predictor.load_state_dict(torch.load(os.path.join(RNDmodel_load_dir, "predictor.pth")))
RNDmodel.target.load_state_dict(torch.load(os.path.join(RNDmodel_load_dir, "target.pth")))

RNDmodel.eval()

num_eval = args.num_eval
# lamda = args.lamda
top_value_size = args.top_value_size
scaling = args.scaling
# top_value_size = Hyperparameters[args.dataset][0]
# scaling = Hyperparameters[args.dataset][1]
# alpha = args.alpha

score_save_path = "test.txt"

env = dataset.env
total_reward = 0.0

env.seed(args.seed)
# value_lst = []

for _ in range(num_eval):

    observation = env.reset()
    episode_reward = 0.

    for t in range(args.max_episode_length):
        
        # if t >= 10:
        #     break

        ## format current observation for conditioning
        conditions = {0: observation}
        _, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
        # value_lst.extend(list(utils.to_np(samples.values)))

        ## uncertainty-value guided planning
        action = uncertainty_value_guide(RNDmodel, samples, scaling, 
                                         "halfcheetah" in args.dataset, int(top_value_size))
        
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        episode_reward += reward
        
        if terminal:
            break

        observation = next_observation
        
    total_reward += episode_reward

# with open(f"scripts/values_re_{args.seed}.txt", "w") as f:
#     for value in value_lst:
#         f.write(f"{value} ")
#     f.write(f"\n{np.array(value_lst).mean()}\n")

score = env.get_normalized_score(total_reward / num_eval)
with open(score_save_path, "a") as f:
    f.write("\n" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + "\n")
    f.write(f"seed: {args.seed}\tnum_eval: {args.num_eval}\n")
    f.write(f"{score}\n")
    
print("\n" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print(f"Robuser\tseed: {args.seed}\tdataset: {args.dataset}")
print(f"normalized score: {score}")

# # total_reward += episode_reward
