import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import time

from robuster.rnd import RNDModel
from robuster.nice import NICE


parser = argparse.ArgumentParser()

parser.add_argument("--visible_gpu", type=str, default="6", help="visible GPU")
parser.add_argument("--keyword", type=str, default="medium", help="the running environment")
parser.add_argument("--num_hidden_units", type=int, default=1200, help="hidden dimension of NICE model")

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

### 2. variable

# rnd
input_size = (4, H)

# output_dim = args.output_dim
if args.keyword == "large":
    output_dim = 384

elif args.keyword == "medium":
    output_dim = 336

elif args.keyword == "umaze":
    output_dim = 240

else:
    raise NameError

save_dir = "RNDmodel/model_{}_{}".format(robuster, args.keyword)

batch_size = 512
n_epochs = 100
learning_rate = 1e-4
horizon_gap = 32
seed = 42

if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

# nice
# input_size = 4 * H
# num_hidden_units = args.num_hidden_units

# # if args.keyword == "large":
# #     num_hidden_units = 1800

# # elif args.keyword == "medium":
# #     num_hidden_units = 1200

# # elif args.keyword == "umaze":
# #     num_hidden_units = 600

# # else:
# #     raise NameError

# save_dir = "robuster/model_{}_{}_{}".format(robuster, args.keyword, num_hidden_units)

# batch_size = 512
# n_epochs = 500
# learning_rate = 1e-4
# horizon_gap = 32
# seed = 42

# if seed:
#     torch.manual_seed(seed)
#     np.random.seed(seed)


### 3. function

def train_rnd():
    
    model = RNDModel(input_size=input_size, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.predictor.parameters(), lr=learning_rate)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
        logger.write("epoch\tloss\n")
    
    orig_dataset = np.load("RNDdataset/" + args.keyword + "_H{}".format(H) + ".npy")
    orig_datasize = orig_dataset.shape
    orig_dataset = np.transpose(orig_dataset.reshape(orig_datasize[0], int(orig_datasize[1]/4), 4), (0, 2, 1))
    orig_datasize = orig_dataset.shape
    print("original dataset size is {}".format(orig_datasize))
    
    # Hindsight truncated plan
    print("Hindsight truncated plan:\nConfiguring...")
    dataset = orig_dataset.copy()
    
    for _ in range(int(H/horizon_gap)):
        
        temp_dataset = orig_dataset.copy()
        res_horizons = np.random.randint(1, int(H/horizon_gap)+1, size=(orig_datasize[0],)) * horizon_gap
        
        for (i, res_horizon) in enumerate(res_horizons):
            if res_horizon != H:
                temp_dataset[i, :, :] = np.concatenate(
                    (temp_dataset[i, :, :1], 
                     temp_dataset[i, :, -(res_horizon-1):], 
                     np.tile(temp_dataset[i, :, :1], (1, int(H-res_horizon)))),
                    axis=1
                )
        dataset = np.concatenate((dataset, temp_dataset))
    
    np.random.shuffle(dataset)
    datasize = dataset.shape
    print("Dataset has been expanded by hindsight truncated plan with the shape of {}".format(datasize))
    
    # res_horizons = torch.randint(1, int(H/horizon_gap)+1, size=(datasize[0],)) * horizon_gap
    # for (i, res_horizon) in enumerate(res_horizons):
    #     if res_horizon != H:
    #         dataset[i, :, :] = torch.concatenate(
    #             (dataset[i, :, :1], dataset[i, :, -(res_horizon-1):], torch.zeros((4, int(H-res_horizon)))), 
    #             dim=1
    #         )
    
    dataloader = DataLoader(TensorDataset(torch.FloatTensor(dataset)), batch_size=batch_size, num_workers=4, 
                            shuffle=True, pin_memory=True)
    
    idxes = np.random.randint(0, datasize[0], size=10000)
    testset = torch.FloatTensor(dataset[idxes]).to(device)

    with torch.no_grad():
        test_loss = model(testset).mean()
        
        print("epoch: 0\tloss: {}".format(test_loss))
        with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
            logger.write("0\t{}\n".format(test_loss))
    
    for epoch in range(n_epochs):
        for batch in tqdm(dataloader):
            
            batch = batch[0].to(device)
            loss = model(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            test_loss = model(testset).mean()
            
            print("epoch: {}\tloss: {}".format(epoch+1, test_loss))
            with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
                logger.write("{}\t{}\n".format(epoch+1, test_loss))
    
    torch.save(model.predictor.state_dict(), os.path.join(save_dir, "predictor.pth"))
    torch.save(model.target.state_dict(), os.path.join(save_dir, "target.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))


# def train_nice():
    
#     model = NICE(data_dim=input_size, num_hidden_units=num_hidden_units).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
#         logger.write("\n{}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
#         logger.write("epoch\tloss\n")
    
#     orig_dataset = np.load("RNDdataset/" + args.keyword + "_H{}".format(H) + ".npy")
#     orig_datasize = orig_dataset.shape
#     print("original dataset size is {}".format(orig_datasize))
    
#     # Hindsight truncated plan
#     print("Hindsight truncated plan:\nConfiguring...")
#     dataset = orig_dataset.copy()
    
#     for _ in range(int(H/horizon_gap)):
        
#         temp_dataset = orig_dataset.copy()
#         res_horizons = np.random.randint(1, int(H/horizon_gap)+1, size=(orig_datasize[0],)) * horizon_gap
        
#         for (i, res_horizon) in enumerate(res_horizons):
#             if res_horizon != H:
#                 temp_dataset[i, :] = np.concatenate(
#                     (temp_dataset[i, :4], 
#                      temp_dataset[i, -4*(res_horizon-1):], 
#                      np.tile(temp_dataset[i, :4], int(H-res_horizon)))
#                 )
#         dataset = np.concatenate((dataset, temp_dataset))
    
#     np.random.shuffle(dataset)
#     datasize = dataset.shape
#     print("Dataset has been expanded by hindsight truncated plan with the shape of {}".format(datasize))
    
#     dataloader = DataLoader(TensorDataset(torch.FloatTensor(dataset)), batch_size=batch_size, num_workers=4, 
#                             shuffle=True, pin_memory=True)
    
#     idxes = np.random.randint(0, datasize[0], size=10000)
#     testset = torch.FloatTensor(dataset[idxes]).to(device)
    
#     with torch.no_grad():
#         test_z, test_log_likelihood = model(testset)
#         test_prior_log_likelihood = torch.sum(model.prior.log_prob(test_z), dim=1).mean()
#         test_loss = -torch.mean(test_log_likelihood)
        
#         print("epoch: 0\tloss: {}\tprior log likelihood: {}".format(test_loss, test_prior_log_likelihood))
#         with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
#             logger.write("epoch: 0\tloss: {}\tprior log likelihood: {}\n".format(test_loss, test_prior_log_likelihood))
    
#     for epoch in range(n_epochs):
#         for batch in tqdm(dataloader):
            
#             batch = batch[0].to(device)
#             _, log_likelihood = model(batch)
#             loss = -torch.mean(log_likelihood)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         with torch.no_grad():
#             test_z, test_log_likelihood = model(testset)
#             test_prior_log_likelihood = torch.sum(model.prior.log_prob(test_z), dim=1).mean()
#             test_loss = -torch.mean(test_log_likelihood)
            
#             print("epoch: {}\tloss: {}\tprior log likelihood: {}".format(epoch+1, test_loss, test_prior_log_likelihood))
#             with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
#                 logger.write("epoch: {}\tloss: {}\tprior log likelihood: {}\n".format(
#                     epoch+1, test_loss, test_prior_log_likelihood))
    
#     torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
#     torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))


if __name__ == "__main__":
    
    ### 4. main
    
    if robuster == "rnd":
        train_rnd()
    
    # if robuster == "nice":
    #     train_nice()
