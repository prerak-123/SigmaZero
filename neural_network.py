import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    """
    The NN Architecture for the agent, consisting of residual layers followed by a policy and value head
    """
    def __init__(self, input_dims=(8,8), input_channels=119, num_hidden_blocks=19, output_dims=4672):
        super(AgentNetwork, self).__init__()
        self.input_dims=input_dims
        self.input_channels=input_channels
        self.num_hidden_blocks=num_hidden_blocks
        self.output_dims=output_dims
        
        self.conv_block = self.get_conv_block(input_channels)
        self.residual_blocks = list()
        for _ in range(num_hidden_blocks):
            self.residual_blocks.append(self.get_residual_block(256))
        self.policy_head = self.get_policy_head(256)
        self.value_head = self.get_value_head(256)
    
    def get_residual_block(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
    
    def get_conv_block(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    
    def get_policy_head(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, self.output_dims),
            nn.Sigmoid()
        )
    
    def get_value_head(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def perform_residual(self, x, ind):
        y = self.residual_blocks[ind](x)
        return F.relu(x + y)
    
    def value_forward(self, x):
        '''
        x: A tensor with shape(N, C, H, W) where N is the batch size, C = input_channels, H,W = input_dims
        '''
        res = self.conv_block(x)
        for i in range(self.num_hidden_blocks):
            res = self.perform_residual(res, i)
            
        res = self.value_head(res)
        return res

    def policy_forward(self, x):
        '''
        x: A tensor with shape(N, C, H, W) where N is the batch size, C = input_channels, H,W = input_dims
        '''
        res = self.conv_block(x)
        for i in range(self.num_hidden_blocks):
            res = self.perform_residual(res, i)
        res = self.policy_head(res)
        return res

    def forward(self, x):
        '''
        x: A tensor with shape(N, C, H, W) where N is the batch size, C = input_channels, H,W = input_dims
        '''
        return (self.value_forward(x), self.policy_forward(x))
        