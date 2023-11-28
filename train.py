# Description: This file contains the training loop for the neural network
# This also includes steps for evaluation and saving the best model

import torch
from torch.nn import Module, MSELoss
import config
import logging
from torch.optim import Adam
import chess.Board
from transform import board_to_input
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from neural_network  import AgentNetwork
from evaluate import Evaluation

from utils import moves_to_output_vector
import os

EPS = 1e-10

## CE Loss for policy head 
def entropy_loss(inpt:torch.Tensor, target:torch.Tensor)->torch.Tensor:
    loss = -torch.mean(torch.sum(target*torch.log(inpt +  EPS), dim = 1))
    return loss
##############################################################################################################

class Trainer:
    
    def __init__(self, model: Module, torch_device = None):
        '''
        Trains the model on the given data
        Reads the data from the memory folder, and trains on it.
        
        model: An instance of AgentNetwork
        torch_device: Sets automatically to the available device if not passed as parameter
        '''
        self.model = model
        self.batch_size = config.BATCH_SIZE
        if(torch_device == None):
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.torch_device = torch_device
        logging.info(f"Training device: {self.torch_device}")  
        
        self.optimiser = Adam(params=self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.value_loss = MSELoss()
        self.policy_loss = entropy_loss
        
    
    def get_Xy(self, data):
        '''
        Read the data from the memory folder and convert it to tensors
        
        data: A list of experiences. Format should be [(state1, winner1, move_probs1), ...]
        
        return: 
        A three tuple (X->Tensor of states, y_value->Tensor of winners, y_policy->Tensor of move probabilities on planes)
        '''
        
        X = torch.stack( [ board_to_input( chess.Board( i[0] ) ) for i in data ] ).to(torch.float32).to(self.torch_device)
        
        y_policy = torch.tensor(np.array([moves_to_output_vector(i[1], chess.Board(i[0])).flatten() for i in data ])).to(torch.float32).to(self.torch_device)
        
        y_value = torch.tensor([ [ i[2] ] for i in data]).to(torch.float32).to(self.torch_device)
        
        return (X, y_value, y_policy)
    
    def train_on_batch(self, X, y_vals, y_probs):
        '''
        Performs a backward step on the input batch
        '''
        
        y_vals_pred, y_probs_pred = self.model(X)
        
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
        
        figure, axis = plt.subplots(2, 2)  

        running_avg_val = np.cumsum(losses[0])/np.arange(1, len(losses[0]) + 1) 
        running_avg_pol = np.cumsum(losses[1])/np.arange(1, len(losses[1]) + 1) 
        

        # for each index calculate cumulative average of last 100 iterations
        axis[0, 0].plot(range(len(losses[0])), losses[0], 'b')
        axis[0, 0].set_title('Value Loss')
        axis[0, 0].set_xlabel('Time Stamp')
        axis[0, 0].set_ylabel('Loss')
        
        axis[1, 0].plot(range(len(running_avg_val)), running_avg_val, 'm')
        axis[1, 0].set_title('Average Value Loss')
        axis[1, 0].set_xlabel('Time Stamp')
        axis[1, 0].set_ylabel('Loss')

        axis[0, 1].plot(range(len(losses[1])), losses[1], 'r')
        axis[0, 1].set_title('Policy Loss')
        axis[0, 1].set_xlabel('Time Stamp')
        axis[0, 1].set_ylabel('Loss')
        
        axis[1, 1].plot(range(len(running_avg_pol)), running_avg_pol, 'g')
        axis[1, 1].set_title('Average Policy Loss')
        axis[1, 1].set_xlabel('Time Stamp')
        axis[1, 1].set_ylabel('Loss')
        
        figure.tight_layout()
                
        plt.savefig(f"{config.IMAGES}loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_LR={config.LEARNING_RATE}.png")
        plt.clf()
        plt.close()
        
    def save_model(self):
        model_str = f"model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
        model_path = f"{config.MODEL}{model_str}"
        torch.save(self.model.state_dict(), model_path)    
        return model_str


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device {device}")

    model = AgentNetwork(input_channels = config.IN_CHANNELS, num_hidden_blocks = config.NUM_BLOCKS).to(device)
    if len(os.listdir(f"{config.BEST_MODEL}")) != 0:
            weight_file = os.listdir(f"{config.BEST_MODEL}")[0]
            model.load_state_dict(torch.load(f"{config.BEST_MODEL}{weight_file}"))
    else:
        torch.save(model.state_dict(), f"{config.BEST_MODEL}best-model.pth")
    
    trainer = Trainer(model)

    while(True):
        experiences = []
        for file in os.listdir(config.MEMORY):
            experiences.extend(np.load(f"./{config.MEMORY}{file}", allow_pickle=True))
        
        losses = trainer.train(experiences, config.TRAIN_STEPS)
        trainer.plot_loss(losses)
        
        new_model = trainer.save_model()
        
        model_eval = Evaluation(f"{config.MODEL}{new_model}", f"{config.BEST_MODEL}best-model.pth")
        
        results = model_eval.evaluate(config.EVAL_GAMES)
        
        if(results["model_1"] >= results["model_2"]):
            print("Obtained a better model")
            os.system(f"cp {config.MODEL}{new_model} {config.BEST_MODEL}best-model.pth")
        
