import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import time
import random

from rnd import RNDModel_halfcheetah, RNDModel_hopper, RNDModel_walker2d


parser = argparse.ArgumentParser()

parser.add_argument("--visible_gpu", type=str, default="0", help="visible GPU")
parser.add_argument("--env", type=str, default="hopper", help="locomotion environment")
parser.add_argument("--level", type=str, default="medium", help="locomotion dataset source")
parser.add_argument("--output_dim", type=int, default=100, help="output dimension of RND model")
parser.add_argument("--seed", type=int, default=1, help="random seed")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters

output_dim = args.output_dim

if args.env == "halfcheetah":
    input_size = 68
    # output_dim = 30
elif args.env == "hopper":
    input_size = (11, 32)
    # output_dim = 150
elif args.env == "walker2d":
    input_size = (17, 32)
    # output_dim = 250
else:
    raise NameError

batch_size = 256
n_epochs = 200
learning_rate = 1e-4

if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

dataset_name = f"{args.env}-{args.level}-v2"
save_dir = f"RNDmodel/{dataset_name}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train_rnd():
    
    if args.env == "halfcheetah":
        model = RNDModel_halfcheetah(input_size=input_size, output_dim=output_dim).to(device)
    elif args.env == "hopper":
        model = RNDModel_hopper(input_size=input_size, output_dim=output_dim).to(device)
    elif args.env == "walker2d":
        model = RNDModel_walker2d(input_size=input_size, output_dim=output_dim).to(device)
    else:
        raise NameError
    
    optimizer = optim.Adam(model.predictor.parameters(), lr=learning_rate)
    
    with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
        logger.write("epoch\tloss\n")
    
    orig_dataset = np.load(os.path.join("RNDdataset", f"{dataset_name}.npy"))
    orig_datasize = orig_dataset.shape
    print("Original dataset size is {}".format(orig_datasize))
    
    if args.env == "halfcheetah":
        dataset = orig_dataset.reshape(orig_datasize[0], -1)
    else:
        dataset = np.transpose(orig_dataset, axes=(0, 2, 1))
    
    datasize = dataset.shape
    print("Dataset size is {}".format(datasize))
    
    dataloader = DataLoader(TensorDataset(torch.FloatTensor(dataset)), batch_size=batch_size, num_workers=1, 
                            shuffle=True, pin_memory=True)
    
    idxes = np.random.randint(0, datasize[0], size=1000)
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

if __name__ == "__main__":
    
    train_rnd()
