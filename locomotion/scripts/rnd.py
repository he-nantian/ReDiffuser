import torch.nn as nn
from torch.nn import init
import numpy as np


class RNDModel_hopper(nn.Module):
    def __init__(self, input_size, output_dim):
        super(RNDModel_hopper, self).__init__()

        self.input_size = input_size    # (11, 32)
        self.output_dim = output_dim

        self.predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=88,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=88,
                out_channels=176,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=176,
                out_channels=176,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=880, out_features=440),
            nn.ReLU(),
            
            nn.Linear(in_features=440, out_features=output_dim),
            nn.ReLU(),
            
            nn.Linear(in_features=output_dim, out_features=output_dim)
        )
        
        self.target = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=88,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=88,
                out_channels=176,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=176,
                out_channels=176,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=880, out_features=output_dim)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv1d):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.dist = nn.PairwiseDistance(p=2)


    # input size is (batch_size, obs_dim, horizon)
    def forward(self, input):

        target_feature = self.target(input)
        predict_feature = self.predictor(input)

        predict_error = self.dist(predict_feature, target_feature)

        return predict_error



class RNDModel_walker2d(nn.Module):
    def __init__(self, input_size, output_dim):
        super(RNDModel_walker2d, self).__init__()

        self.input_size = input_size    # (17, 32)
        self.output_dim = output_dim

        self.predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=136,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=136,
                out_channels=272,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=272,
                out_channels=272,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=1360, out_features=680),
            nn.ReLU(),
            
            nn.Linear(in_features=680, out_features=output_dim),
            nn.ReLU(),
            
            nn.Linear(in_features=output_dim, out_features=output_dim)
        )
        
        self.target = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=136,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=136,
                out_channels=272,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=272,
                out_channels=272,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=1360, out_features=output_dim)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv1d):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.dist = nn.PairwiseDistance(p=2)


    # input size is (batch_size, obs_dim, horizon)
    def forward(self, input):

        target_feature = self.target(input)
        predict_feature = self.predictor(input)

        predict_error = self.dist(predict_feature, target_feature)

        return predict_error



class RNDModel_halfcheetah(nn.Module):
    def __init__(self, input_size, output_dim):
        super(RNDModel_halfcheetah, self).__init__()

        self.input_size = input_size    # 68
        self.output_dim = output_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, output_dim)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.dist2 = nn.PairwiseDistance(p=2)


    def forward(self, input):

        target_feature = self.target(input)
        predict_feature = self.predictor(input)

        predict_error = self.dist2(predict_feature, target_feature)

        return predict_error
