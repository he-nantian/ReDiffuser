import numpy as np
import os
import torch
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

UMAZE_RANGE = np.array([[0.5, 3.5], [0.5, 3.5]])
MEDIUM_RANGE = np.array([[0.5, 6.5], [0.5, 6.5]])
LARGE_RANGE = np.array([[0.5, 7.5], [0.5, 10.5]])

## Hyperparameters
keyword = "large"
state_range = LARGE_RANGE
data_size = 1024 ** 2
batch_size = 16
H = 384
T = 256

# keyword = "medium"
# state_range = MEDIUM_RANGE
# data_size = 1024 * 512
# batch_size = 16
# H = 256
# T = 256

# keyword = "umaze"
# state_range = UMAZE_RANGE
# data_size = 512 ** 2
# batch_size = 16
# H = 128
# T = 64

seed = 0
print("Random seed: {}".format(seed))
torch.manual_seed(seed)
np.random.seed(seed)


def generate_dataset():
    
    diffusion_experiment = utils.load_diffusion("logs", "maze2d-{}-v1".format(keyword), 
                                                "diffusion/H{}_T{}".format(H, T), epoch="latest")
    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    policy = Policy(diffusion, dataset.normalizer)
    
    n_iterations = int(data_size / batch_size)
    print("Generating iterations: {}".format(n_iterations))
    
    RNDdataset = np.empty(shape=(0, 4 * H))
    
    x_bound = state_range[0, 1] - state_range[0, 0]
    x_offset = state_range[0, 0]
    y_bound = state_range[1, 1] - state_range[1, 0]
    y_offset = state_range[1, 0]
    print("x_bound: {}, x_offset: {}, y_bound: {}, y_offset: {}".format(x_bound, x_offset, y_bound, y_offset))

    for i in range(n_iterations):
        
        initial_x = np.random.random((1,)) * x_bound + x_offset
        target_x = np.random.random((1,)) * x_bound + x_offset
        initial_y = np.random.random((1,)) * y_bound + y_offset
        target_y = np.random.random((1,)) * y_bound + y_offset
        initial_vel = np.random.random((2,)) * 2 - 1
        initial_state = np.concatenate((initial_x, initial_y, initial_vel))
        target_state = np.concatenate((target_x, target_y, np.array([0., 0.])))
        
        cond = {}
        cond[0] = initial_state
        cond[H-1] = target_state
        
        _, trajectories = policy(cond, batch_size=batch_size)
        states = trajectories.observations    # array: batch_size * H * 4
        
        # keep the target state before the initial state
        states = np.concatenate((states[:, -1:, :], states[:, :-1, :]), axis=1).reshape(batch_size, -1)
        RNDdataset = np.concatenate((RNDdataset, states))
        
        if (i+1) % 16 == 0:
            print("{}%  ----  Generate dataset of size:  {}/{}".format(
                int(100*(i+1)*batch_size/data_size), (i+1)*batch_size, data_size))
    
    np.random.shuffle(RNDdataset)
    np.save("RNDdataset/" + keyword + "_H{}".format(H) + ".npy", RNDdataset)
    print("RNDdataset has been shuffled and saved with shape of {}".format(RNDdataset.shape))


if __name__ == "__main__":
    generate_dataset()