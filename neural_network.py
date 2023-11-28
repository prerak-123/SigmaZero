# Description: The neural network architecture for the agent

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################################################
# Helper functions for the architecture
def get_conv_block(in_channels:int, out_channels:int=256)->nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    """
    Residual block for the architecture
    """
    def __init__(self,in_channels:int=256, mid_channels:int=256, out_channels:int=256):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x:torch.Tensor):
        return F.relu(x + self.block(x))
##############################################################################################################

class AgentNetwork(nn.Module):
    """
    The NN Architecture for the agent, consisting of residual layers followed by a policy and value head
    """
    def __init__(self, input_dims:(int,int)=(8,8), input_channels:int=119, num_hidden_blocks:int=19, output_dims:int=4672):
        super(AgentNetwork, self).__init__()
        self.input_dims=input_dims
        self.input_channels=input_channels
        self.num_hidden_blocks=num_hidden_blocks
        self.output_dims=output_dims
        
        self.encoder = self.get_encoder(input_channels=input_channels,num_res_blocks=num_hidden_blocks)
                                     
        self.policy_head = self.get_policy_head(input_channels=256)
        self.value_head = self.get_value_head(input_channels=256)
    
    def get_encoder(self, input_channels:int,num_res_blocks:int)->nn.Sequential:
        enc = nn.Sequential(get_conv_block(in_channels=input_channels, out_channels=256))
        for i in range(num_res_blocks):
            enc.add_module(f'ResBlock_{i}', ResBlock())
        return enc
        
    def get_policy_head(self, input_channels:int)->nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, self.output_dims),
            nn.Softmax(dim=1)
        )
    
    def get_value_head(self, input_channels:int)->nn.Sequential:
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
        
        
    '''
    x: A tensor with shape(N, C, H, W) where N is the batch size, C = input_channels, H,W = input_dims
    '''
    # for just the value forward pass
    def value_forward(self, x:torch.Tensor)->torch.Tensor:
        emb = self.encoder(x)
        return self.value_head(emb)
    #for just the policy forward pass
    def policy_forward(self, x:torch.Tensor)->torch.Tensor:
        emb = self.encoder(x)
        return self.policy_head(emb)

    #for both the value and policy forward pass / this is the forward pass for the entire network
    def forward(self, x:torch.Tensor)->(torch.Tensor, torch.Tensor):
        emb = self.encoder(x)
        return (self.value_head(emb), self.policy_head(emb))
        
##############################################################################################################