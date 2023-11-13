import pandas as pd
import config
import threading

from agent import Agent
from chessEnv import ChessEnv
from game import Game

from chess import Move

CSV_FILE = "puzzles.csv"
N = 2

curr_index = 0
lock = threading.Lock()

def play_puzzle(fen, moves):
    white = Agent(local_preds=True)
    black = Agent(local_preds=True)
    
    env = ChessEnv(fen)
    env.fen = env.step(Move.from_uci(moves[0])).fen()
    
    game = Game(env, white, black)
    
    game.game()

def thread_func(dataset):
    global curr_index
    global lock
    while(True):
        with lock:
            if(curr_index >= len(dataset)):
                return
            my_index = curr_index
            print(f"My Index {my_index}")
            curr_index += 1
        fen, moves = dataset["FEN"][my_index], dataset["Moves"][my_index].split()
        
        play_puzzle(fen, moves)
        

if __name__ == "__main__":
    dataset = pd.read_csv(f"{config.PUZZLE}{CSV_FILE}")
    dataset = dataset[["FEN", "Moves", "Rating"]]
    dataset["Rating"] = pd.to_numeric(dataset["Rating"])
    dataset.sort_values(by="Rating", inplace=True)
    
    threads = []
    for i in range(N):
        t = threading.Thread(target=thread_func, args=(dataset,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
