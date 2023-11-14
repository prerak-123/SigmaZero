import pandas as pd
import config
import threading

from agent import Agent
from chessEnv import ChessEnv
from game import Game
from multiprocessing import Pool
from chess import Move

CSV_FILE = "puzzles.csv"
N = 5

def play_puzzle(fen, moves):
    white = Agent(local_preds=True)
    black = Agent(local_preds=True)
    
    env = ChessEnv(fen)
    env.fen = env.step(Move.from_uci(moves[0])).fen()
    
    game = Game(env, white, black)
    
    game.game()

def thread_func(i):
    global dataset
    if(i >= len(dataset)):
        return
    my_index = i
    while True:
        print(f"My Index {my_index}")
        fen, moves = dataset["FEN"][my_index], dataset["Moves"][my_index].split()       
        play_puzzle(fen, moves)
        my_index += N
        

if __name__ == "__main__":
    dataset = pd.read_csv(f"{config.PUZZLE}{CSV_FILE}")
    dataset = dataset[["FEN", "Moves", "Rating"]]
    dataset["Rating"] = pd.to_numeric(dataset["Rating"])
    dataset.sort_values(by="Rating", inplace=True)
   
    with Pool() as p:
        p.map(thread_func, range(0, N))