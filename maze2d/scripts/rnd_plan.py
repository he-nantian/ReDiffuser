import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import os
import time
import math

from robuster.rnd import RNDModel
from robuster.nice import NICE

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


parser = argparse.ArgumentParser()

parser.add_argument("--visible_gpu", type=str, default="0", help="visible GPU")
parser.add_argument("--keyword", type=str, default="large", help="the running environment")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--n_plans", type=int, default=10, help="how many times to interface with env")
parser.add_argument("--use_robuster", action="store_true", help="whether using robuster to plan")
parser.add_argument("--prob_sample", action="store_true", default=False, help="sample plan with softmax")
# parser.add_argument("--discount_power", type=float, default=0.0, help="how much we doubt at shorter horizons")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters

### 1. robuster

robuster = "rnd"
# robuster = "nice"

if args.keyword == "large":
    H = 384
    T = 256

elif args.keyword == "medium":
    H = 256
    T = 256

elif args.keyword == "umaze":
    H = 128
    T = 64

else:
    raise NameError

### 2. specific variables

# rnd
input_size = (4, H)

if args.keyword == "large":
    output_dim = 384
    discount_power = 0.6

elif args.keyword == "medium":
    output_dim = 336
    discount_power = 0.4

elif args.keyword == "umaze":
    output_dim = 240
    discount_power = 0.2

else:
    raise NameError

# discount_power = args.discount_power
load_dir = "RNDmodel/model_{}_{}".format(robuster, args.keyword)
save_dir = "test_{}".format(args.keyword)

# nice
# input_size = 4 * H
# # discount_power = args.discount_power

# if args.keyword == "large":
#     num_hidden_units = 1600
#     discount_power = 1.0

# elif args.keyword == "medium":
#     num_hidden_units = 1500
#     discount_power = 0.6

# elif args.keyword == "umaze":
#     num_hidden_units = 1000
#     discount_power = 0.0

# else:
#     raise NameError

# if args.use_robuster:
#     if args.prob_sample:
#         tail_name = "prob_sample"
#     else:
#         tail_name = "dir_choose"
# else:
#     tail_name = "random"

# load_dir = "robuster/model_{}_{}".format(robuster, args.keyword)
# save_dir = "robuster/score_{}_{}_{}".format(robuster, args.keyword, tail_name)

# common variables
n_plans = args.n_plans
n_samples = 16
horizon_gap = 32

seed = args.seed
if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

possible_Hs = np.arange(horizon_gap, H+1, horizon_gap)


def soft_max(a):
    
    assert len(a.shape) == 1
    a -= np.max(a)
    
    return np.exp(a) / np.sum(np.exp(a))


# (n, discount_H, 4)
def fit_model(n_states, horizon, pad=False):
    
    n_states_shape = n_states.shape
    
    # move target state to the first position
    n_fit_states = np.concatenate((n_states[:, -1:, :], n_states[:, :-1, :]), axis=1)
    
    if pad:
        n_fit_states = np.concatenate(
            (n_fit_states, np.tile(n_fit_states[:, :1, :], (1, int(horizon-n_states_shape[1]), 1))),
            axis=1
        )
    
    ### 3. fit data to robuster: (n, H, 4)
    
    # rnd: (n, 4, H)
    n_fit_states = np.transpose(n_fit_states, (0, 2, 1))
    
    # nice: (n, 4*H)
    # n_fit_states = n_fit_states.reshape(n_fit_states.shape[0], -1)
    
    # ntc: (n, 4*H)
    # n_fit_states = n_fit_states.reshape(n_fit_states.shape[0], -1)
    
    return n_fit_states


def revert4plan(fit_states):
        
    ### 4. revert data for planning: --> (H, 4)
    
    # rnd: (4, H)
    reverted_states = np.transpose(fit_states, axes=(1, 0))
    
    # nice: (4*H, )
    # reverted_states = fit_states.reshape(H, 4)
    
    # ntc: (4*H, )
    # reverted_states = reverted_states.reshape(H, 4)
    
    # move target state to the last position
    reverted_states = np.concatenate((reverted_states[1:], reverted_states[:1]))
    
    return reverted_states


# shape is fit to robuster, e.g. rnd: (n, 4, H)
def choose_trajectory(n_fit_states, model, discount=1.0):
    
    inputs = torch.FloatTensor(n_fit_states).to(device)
    
    ### 5. convert model output to uncertainty
    
    # rnd
    n_uncertainty = model(inputs).detach().cpu().data.numpy().flatten() / discount
    
    # nice
    # z, _ = model(inputs)
    # n_uncertainty = (- torch.sum(model.prior.log_prob(z), dim=1)).detach().cpu().data.numpy().flatten() / discount
    
    # ntc
    # n_likelihoods = model(inputs)["likelihoods"]["y"]
    # n_entropy = - torch.sum(torch.log(n_likelihoods), dim=1) / math.log(2)
    # n_uncertainty = n_entropy.detach().cpu().data.numpy().flatten() / discount
    
    if args.prob_sample:
        fit_states = None
        
        probs = soft_max(-n_uncertainty)
        
        sample_prob = np.random.random()
        cum_prob = 0.
        
        for (prob, fit_states) in zip(probs, n_fit_states):
            
            cum_prob += prob
            if sample_prob < cum_prob:
                break
        return fit_states, n_uncertainty.mean()
    
    else:
        min_idx = np.argmin(n_uncertainty)
        return n_fit_states[min_idx], n_uncertainty.mean()


def robust_plan():
    
    env = datasets.load_environment("maze2d-{}-v1".format(args.keyword))
    if seed:
        env.seed(seed) # type: ignore

    diffusion_experiment = utils.load_diffusion("logs", "maze2d-{}-v1".format(args.keyword), 
                                                "diffusion/H{}_T{}".format(H, T), epoch="latest")
    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    policy = Policy(diffusion, dataset.normalizer)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(os.path.join(save_dir, "{}_{}_{}.txt".format(n_plans, n_samples, horizon_gap)), "a") as log_f:
        log_f.write("\n{}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        log_f.write("random seed: {}\nusing robuster: {}\n".format(seed, args.use_robuster))
    
    ### 6. load model
    
    # rnd
    model = RNDModel(input_size=input_size, output_dim=output_dim).to(device)
    model.target.load_state_dict(torch.load(os.path.join(load_dir, "target.pth")))
    model.predictor.load_state_dict(torch.load(os.path.join(load_dir, "predictor.pth")))
    
    # nice
    # model = NICE(input_size, num_hidden_units=num_hidden_units).to(device)
    # model.load_state_dict(torch.load(os.path.join(load_dir, "best_model.pth")))
    
    # ntc
    # model = NTCModel(input_dim, hidden_dim, compressed_dim).to(device)
    # model.load_state_dict(torch.load(os.path.join(load_dir, "model.pth")))
    
    model.eval()
    
    score_mean = 0.
    for plan_id in range(n_plans):

        observation = env.reset() # type: ignore
        # env.set_target() # type: ignore
        target = env._target # type: ignore
        
        cond = {
            0: observation,
            H-1: np.array([*target, 0, 0])
        }
        
        if not args.use_robuster:
            _, samples = policy(cond, batch_size=1)
            plan_states = samples.observations[0]
            # print("{}: Not using robuster: horizon is fixed as {}".format(plan_id + 1, H))
        
        else:
            best_states = None
            lowest_uncertainty = np.inf
            best_horizon = 0
            
            for horizon in reversed(possible_Hs):
                cond = {
                    0: observation,
                    horizon - 1: np.array([*target, 0, 0])
                }
                
                policy.horizon = horizon
                _, samples = policy(cond, batch_size=n_samples)
                n_states = samples.observations
                
                # rendering the original planning trajectory with horizon H
                renderer.composite(os.path.join(save_dir, f"{plan_id}_h{horizon}.png"), samples.observations[0][None], ncol=1)
                
                n_fit_states = fit_model(n_states, horizon=H, pad=(horizon < H))
                fit_states, uncertainty = choose_trajectory(n_fit_states, model, discount=(horizon/H)**discount_power)
                
                if uncertainty < lowest_uncertainty:
                    best_states = revert4plan(fit_states)
                    best_horizon = horizon
                    lowest_uncertainty = uncertainty
            # print("{}: Best horizon has been chosen with {}".format(plan_id + 1, best_horizon))
            plan_states = best_states
        
        plan_states = np.array(plan_states)
        
        # rendering the chosen planning trajectory with horizon h
        renderer.composite(os.path.join(save_dir, f"{plan_id}_h{best_horizon}.png"), plan_states[None], ncol=1)

        total_reward = 0
        for t in range(env.max_episode_steps): # type: ignore

            state = env.state_vector().copy() # type: ignore

            if t < len(plan_states) - 1:
                next_waypoint = plan_states[t+1]
            else:
                next_waypoint = plan_states[-1].copy()
                next_waypoint[2:] = 0
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = env.step(action) # type: ignore
            total_reward += reward
            
            if terminal:
                break

            observation = next_observation

        score = env.get_normalized_score(total_reward) # type: ignore
        score_mean += score
    
    score_mean /= n_plans
    print("mean score is {}".format(score_mean))
    with open(os.path.join(save_dir, "{}_{}_{}.txt".format(n_plans, n_samples, horizon_gap)), "a") as log_f:
        log_f.write("prob_sample: {}\n".format(args.prob_sample) + str(score_mean) + "\n")


if __name__ == "__main__":
    
    robust_plan()
