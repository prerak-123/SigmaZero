import numpy as np
import torch
import torch.nn as nn
import config
import logging
from torch.optim import Adam
import chess
from transform import board_to_input
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class Trainer:
    
    def __init__(self, model: nn.Module, torch_device = None):
        '''
        model: An instance of AgentNetwork
        torch_device: Sets automatically to the available device if not passed as parameter
        '''
        self.model = model
        self.batch_size = config.BATCH_SIZE
        if(torch_device == None):
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.torch_device = torch_device
        logging.info(f"Training device: {self.torch_device}")  
        
        self.optimiser = Adam(params=model.parameters(), lr=config.LEARNING_RATE)
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
        
    
    def get_Xy(self, data):
        '''
        data: A list of experiences. Format should be [(state1, winner1, move_probs1), ...]
        
        return: 
        A three tuple (X->Tensor of states, y_value->Tensor of winners, y_policy->Tensor of move probabilities on planes)
        '''
        
        #TODO: Change the function used here, or modify it
        X = torch.cat([board_to_input(chess.Board(i[0])) for i in data]).to(torch.float32).to(self.torch_device)
        
        y_value = torch.tensor([ [ i[1] ] for i in data]).to(torch.float32).to(self.torch_device)
        
        y_policy = torch.tensor(np.array([i[2] for i in data ])).to(torch.float32).to(self.torch_device)
        
        return (X, y_value, y_policy)
    
    def train_on_batch(self, X, y_vals, y_probs):
        '''
        Performs a backward step on the input batch
        '''
        
        y_vals_pred, y_probs_pred = self.model(X)
        
        #TODO: Could add tunable lambda
        val_loss = self.value_loss(y_vals_pred, y_vals)
        policy_loss =  self.policy_loss(y_probs_pred, y_probs)
        loss = val_loss + policy_loss
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return val_loss.item(), policy_loss.item()
    
    def train(self, data, steps):
        losses = [[], []]
        
        X, y_value, y_policy = self.get_Xy(data)
        for _ in tqdm(range(steps)):
            indexes = np.random.choice(len(data), size=self.batch_size, replace=True)
            X_batch = X[indexes]
            y_val_batch = y_value[indexes]
            y_policy_batch = y_policy[indexes]
            
            val_loss, policy_loss = self.train_on_batch(X_batch, y_val_batch, y_policy_batch)
            losses[0].append(val_loss)
            losses[1].append(policy_loss)
            
        return losses

    def plot_loss(self, losses):
        plt.subplot(1, 2, 1)  
        plt.plot(range(len(losses[0])), losses[0], 'b')
        plt.title('Value Loss')
        plt.xlabel('Time Stamp')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        
        plt.plot(range(len(losses[1])), losses[1], 'r')
        plt.title('Policy Loss')
        plt.xlabel('Time Stamp')
        plt.ylabel('Loss')
        
        plt.tight_layout(4)
        
        plt.savefig(f"{config.IMAGES}loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")
        
    def save_model(self):
        torch.save(self.model, f"{config.MODEL}model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth")    

'''
TODO:
1. Decide if this file is run individually or called by main.py
2. Decide on format of storing and restoring experiences
'''