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
        # print("in agent run sims")
        # self.model.eval()
        # print('after eval')
        with torch.no_grad():
            # print("hello")
            # print(self.mcts)
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
  
## reference - to remove      
'''
class __Agent:
    def __init__(self, local_predictions: bool = False, model_path = None, state=chess.STARTING_FEN):
        
        if local_predictions and model_path is not None:
            logging.info("Using local predictions")
            from tensorflow.python.ops.numpy_ops import np_config
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            self.local_predictions = True
            np_config.enable_numpy_behavior()
        else:
            logging.info("Using server predictions")
            self.local_predictions = False
            # connect to the server to do predictions
            try: 
                self.socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_to_server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                server = os.environ.get("SOCKET_HOST", "localhost")
                port = int(os.environ.get("SOCKET_PORT", 5000))
                self.socket_to_server.connect((server, port))
            except Exception as e:
                print(f"Agent could not connect to the server at {server}:{port}: ", e)
                exit(1)
            logging.info(f"Agent connected to server {server}:{port}")

        self.mcts = MCTS(self, state=state)
        

    def build_model(self) -> Model:
        """
        Build a new model based on the configuration in config.py
        """
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model

    def run_simulations(self, n: int = 1):
        """
        Run n simulations of the MCTS algorithm. This function gets called every move.
        """
        print(f"Running {n} simulations...")
        self.mcts.run_simulations(n)

    def save_model(self, timestamped: bool = False):
        """
        Save the current model to a file
        """
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    def predict(self, data):
        """
        Predict locally or using the server, depending on the configuration
        """
        if self.local_predictions:
            # use tf.function
            import local_prediction
            p, v = local_prediction.predict_local(self.model, data)
            return p.numpy(), v[0][0]
        return self.predict_server(data)

    def predict_server(self, data: np.ndarray):
        """
        Send data to the server and get the prediction
        """
        # send data to server
        self.socket_to_server.send(f"{len(data.flatten()):010d}".encode('ascii'))
        self.socket_to_server.send(data)
        # get msg length
        data_length = self.socket_to_server.recv(10)
        data_length = int(data_length.decode("ascii"))
        # get prediction
        response = utils.recvall(self.socket_to_server, data_length)
        # decode response
        response = response.decode("ascii")
        # json to dict
        response = json.loads(response)
        # unpack dictionary to tuple
        return np.array(response["prediction"]), response["value"]
'''
if __name__=="__main__":
    pass