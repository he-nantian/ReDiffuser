import torch.nn as nn
from torch.nn import init
import numpy as np


class RNDModel(nn.Module):
    def __init__(self, input_size, output_dim):
        super(RNDModel, self).__init__()

        self.input_size = input_size    # (4, H)
        self.output_dim = output_dim

        self.predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=64,
                kernel_size=4,
                stride=4
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.AdaptiveAvgPool1d(output_size=7),
            nn.Flatten(),
            
            nn.Linear(in_features=1792, out_features=896),
            nn.ReLU(),
            
            nn.Linear(in_features=896, out_features=output_dim),
            nn.ReLU(),
            
            nn.Linear(in_features=output_dim, out_features=output_dim)
        )
        
        self.target = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=64,
                kernel_size=4,
                stride=4
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2
            ),
            nn.LeakyReLU(),
            
            nn.AdaptiveAvgPool1d(output_size=7),
            nn.Flatten(),
            
            nn.Linear(in_features=1792, out_features=output_dim)
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
        
        self.dist2 = nn.PairwiseDistance(p=2)

    # input size is (N, 4, L)
    def forward(self, input):

        target_feature = self.target(input)
        predict_feature = self.predictor(input)

        predict_error = self.dist2(predict_feature, target_feature)

        return predict_error