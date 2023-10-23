import numpy as np
import socket
import chess
from chess import Move, PieceType
import time
import config

### the following two will be integrated when the API is ready
from node import Node
from mapper import Mapping

from PIL import Image
def save_state_to_image(state:np.ndarray,path:str):
    st = time.time()
    state = np.pad((state*np.uint8(255)),((0, 0), (1, 1), (1, 1)),'constant', constant_values=128)
    full_array = np.pad(np.concatenate(state, axis=1) , ((4, 4), (5, 5)),'constant', constant_values=128)
    img = Image.fromarray(full_array)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(path)
    print(f'Image saved to {path} in {(time.time()-st):.6f} seconds')
    
def timeit(method):
    def timed(*args, **kw):
        st = time.time()
        result = method(*args, **kw)
        et = time.time()
        print(f'{method.__name__} took {(et-st) :.4s} seconds')
        return result
    return timed

def recvall(sock: socket.socket, count: int = 0) -> bytes:
    """
    Function to continuously receive data from a socket
    """
    buffer = b''
    if count == 0:
        while True:
            part = sock.recv(config.SOCKET_BUFFER_SIZE)
            buffer += part
            if len(part) < config.SOCKET_BUFFER_SIZE:
                break
    else:
        while count > 0:
            part = sock.recv(config.SOCKET_BUFFER_SIZE)
            buffer += part
            count -= len(part)
    return buffer


### the following will be integrated into env maybe when the API is ready

def moves_to_output_vector(moves:dict,board:chess.Board):
    pass
def move_to_plane_index(move:str,board:chess.Board):
    pass


if __name__=="__main__":
    pass
