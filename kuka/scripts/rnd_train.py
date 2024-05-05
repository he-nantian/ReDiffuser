import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import time
import random

from rnd import RNDModel


parser = argparse.ArgumentParser()

parser.add_argument("--visible_gpu", type=str, default="7", help="visible GPU")
parser.add_argument("--keyword", type=str, default="unconditional", help="kuka task")
# parser.add_argument("--output_dim", type=int, default=510, help="output dimension of RND model")
parser.add_argument("--seed", type=int, default=0, help="seed for RND model training")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters

h = 96
H = 128
T = 100
keyword = args.keyword
output_dim = 510


input_size = (39, H)

batch_size = 256
n_epochs = 100
learning_rate = 1e-4
horizon_gap = 8
seed = args.seed

if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

save_dir = "rnd_{}_{}_{}".format(keyword, output_dim, seed)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train_rnd():
    
    model = RNDModel(input_size=input_size, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.predictor.parameters(), lr=learning_rate)
    
    with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
        logger.write("epoch\tloss\n")
    
    orig_dataset = np.load(os.path.join("dataset_{}_{}".format(H, T), "{}.npy".format(keyword)))
    orig_datasize = orig_dataset.shape
    print("Original dataset size is {}".format(orig_datasize))
    
    # Hindsight truncated plan
    print("Hindsight truncated plan:\nConfiguring...")
    dataset = orig_dataset.copy()
    
    for _ in range(int((H - h) / horizon_gap) + 1):
        
        temp_dataset = orig_dataset.copy()
        res_horizons = np.random.randint(int(h / horizon_gap), int(H / horizon_gap) + 1, size=(orig_datasize[0],)) * horizon_gap
        
        for (i, res_horizon) in enumerate(res_horizons):
            if res_horizon != H:
                discard_idxes = random.sample(range(1, H-1), H-res_horizon)
                temp_dataset[i, :, :] = np.concatenate(
                    (np.delete(temp_dataset[i, :, :], discard_idxes, axis=1), 
                     np.tile(temp_dataset[i, :, -1:], (1, int(H-res_horizon)))),
                    axis=1
                )
        dataset = np.concatenate((dataset, temp_dataset))
    
    np.random.shuffle(dataset)
    datasize = dataset.shape
    print("Dataset has been expanded by hindsight truncated plan with the shape of {}".format(datasize))
    
    dataloader = DataLoader(TensorDataset(torch.FloatTensor(dataset)), batch_size=batch_size, num_workers=4, 
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
