import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils

import os
import time
import numpy as np


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'
    visible_gpu: str = '4'
    seed: int = 8
    num_eval: int = 10

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

num_eval = args.num_eval

env = dataset.env
total_reward = 0.

env.seed(args.seed)
# value_lst = []

for eval_id in range(num_eval):

    observation = env.reset()
    episode_reward = 0.

    for t in range(args.max_episode_length):
        
        # if t >= 10:
        #     break

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
        # value_lst.extend(list(utils.to_np(samples.values)))

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        episode_reward += reward
        
        if terminal:
            break

        observation = next_observation
        
    total_reward += episode_reward
        
score = env.get_normalized_score(total_reward / num_eval)

# with open(f"scripts/values_{args.seed}.txt", "w") as f:
#     for value in value_lst:
#         f.write(f"{value} ")
#     f.write(f"\n{np.array(value_lst).mean()}\n")


with open("test.txt", "a") as f:
    f.write("\n" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + "\n")
    f.write(f"seed: {args.seed}\tnum_eval: {args.num_eval}\n")
    f.write(f"normalized score: {score}\n")

print("\n" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print(f"Diffuser\tseed: {args.seed}\tdataset: {args.dataset}")
print(f"normalized score: {score}")