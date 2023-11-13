import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_network import AgentNetwork
# from mcts import MCTS # missing
from CPP_backend import *
import utils
import chess
import time
import config
import datetime

class Agent:
    def __init__(self,local_preds:bool = False, model_path:str|None = None,state:str = chess.STARTING_FEN):
        """
        An agent is an object that can play chessmoves on the environment.
        Based on the parameters, it can play with a local model, or send its input to a server.
        It holds an MCTS object that is used to run MCTS simulations to build a tree.
        """
        self.local_preds = local_preds
        if local_preds:
            self.model = AgentNetwork(input_channels = config.IN_CHANNELS, num_hidden_blocks = config.NUM_BLOCKS)
            if model_path is not None:
                self.model.load_state_dict(torch.load(model_path))
        else :
            raise NotImplementedError("Server predictions not implemented yet")
        
        self.state = state
        self.mcts = MCTS(self, state, True)
        
    def run_simulations(self,n:int=1):
        with torch.no_grad():

            self.mcts.run_simulations(n)
            
    def save_model(self,timestamped:bool = False)->str:
        if timestamped:
            model_path = f"{config.MODEL}model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
            
            torch.save(self.model.state_dict(), model_path)
        else:
            model_path = f"{config.MODEL}model.pth"
            torch.save(self.model.state_dict(), model_path)
        
        return model_path
            
    def predict(self, data:torch.Tensor):
        data = torch.Tensor(data).to(torch.float32).unsqueeze(0)
        # print(data.shape)
        # print("in agent predict")
        if self.local_preds:
            # print('local')
            return self.predict_local(data)
        return self.predict_server(data)
    
    def predict_local(self,data:torch.Tensor):
        # self.model.eval()
        
        with torch.no_grad():
            v, p = self.model(data)
            return p, v.item()

    def predict_server(self,data:torch.Tensor):
        raise NotImplementedError("Server predictions not implemented yet")
  
if __name__=="__main__":
    pass