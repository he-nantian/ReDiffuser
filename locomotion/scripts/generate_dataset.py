import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils

import os
import time
import numpy as np
from tqdm import tqdm


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'
    visible_gpu: str = '4'
    seed: int = 1
    num_data: int = 64 ** 2    # 256 ** 2
    num_candidates_per_plan: int = 64

args = Parser().parse_args('plan')

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu


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

num_data = args.num_data
num_iters = int(num_data / 4)
num_candidates_per_plan = args.num_candidates_per_plan

env = dataset.env
env.seed(args.seed)

trajectory_dataset = np.zeros(shape=(num_data, args.horizon, diffusion.observation_dim))
# transition_dataset = np.zeros(shape=(num_data, diffusion.transition_dim))

for iter_id in tqdm(range(num_iters)):

    observation = env.reset()
    conditions = {0: observation}
    _, samples = policy(conditions, batch_size=num_candidates_per_plan, horizon=None, verbose=args.verbose)
    
    # obs_plan = samples.observations[0]
    
    # first_transition = np.concatenate((observation, samples.actions[0, 0]))
    
    trajectory_dataset[iter_id*4:(iter_id+1)*4] = samples.observations[:4]
    # transition_dataset[iter_id] = first_transition
    
idxes = np.arange(num_data)
np.random.shuffle(idxes)
trajectory_dataset = trajectory_dataset[idxes]
# transition_dataset = transition_dataset[idxes]

np.save(os.path.join("RNDdataset", f"{args.dataset}.npy"), trajectory_dataset)
# np.save(os.path.join("transition_dataset", f"{args.dataset}.npy"), transition_dataset)

print("trajectory dataset has been shuffled and saved with shape of {}".format(trajectory_dataset.shape))
# print("transition dataset has been shuffled and saved with shape of {}".format(transition_dataset.shape))