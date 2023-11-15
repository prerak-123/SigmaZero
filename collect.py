import pandas as pd
import config
import threading

from agent import Agent
from chessEnv import ChessEnv
from game import Game
from multiprocessing import Pool
from chess import Move

import sys
import os
import numpy as np

CSV_FILE = "puzzles.csv"
N = 5

def play_puzzle(fen, moves):
    model_path = None if len(os.listdir(config.BEST_MODEL)) == 0 else f"{config.BEST_MODEL}best-model.pth"
    white = Agent(local_preds=True, model_path=model_path)
    black = Agent(local_preds=True, model_path=model_path)
    
    env = ChessEnv(fen)
    env.fen = env.step(Move.from_uci(moves[0])).fen()
    
    game = Game(env, white, black)
    
    game.game()

def play_normal():
    model_path = None if len(os.listdir(config.BEST_MODEL)) == 0 else f"{config.BEST_MODEL}best-model.pth"
    white=Agent(local_preds=True, model_path=model_path)
    black=Agent(local_preds=True, model_path=model_path)

    env = ChessEnv()
    game = Game(env, white, black)

    game.game()

def thread_func(i):
    np.random.seed(i)
    global dataset
    my_index = i
    while True:
        if(np.random.random() < config.PUZZLE_PROB):

            if(my_index >= len(dataset)):
                return
            print(f"Playing puzzle of index {my_index}")
            fen, moves = dataset["FEN"][my_index], dataset["Moves"][my_index].split()       
            play_puzzle(fen, moves)
            my_index += N
        
        else:
            print(f"Playing normal game")
            play_normal()
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage python3 collect.py <last_game>")
        sys.exit()
    
    last_game = int(sys.argv[1])

    dataset = pd.read_csv(f"{config.PUZZLE}{CSV_FILE}")
    dataset = dataset[["FEN", "Moves", "Rating"]]
    dataset["Rating"] = pd.to_numeric(dataset["Rating"])
    dataset.sort_values(by="Rating", inplace=True)
   
    with Pool() as p:
        p.map(thread_func, range(last_game, last_game + N))
