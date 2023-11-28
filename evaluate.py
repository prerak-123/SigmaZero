# Evaluates the performance of two models against each other
# Used to find if the model is improving
from agent import Agent
from chessEnv import ChessEnv
import chess.pgn
import config
from game import Game
from random import shuffle

def get_openings(n: int, pgn_file: str = f"{config.PGN}Openings_sf.pgn"):
    
    with open(pgn_file) as pgn:
        openings = []
        game = chess.pgn.read_game(pgn)
        while game is not None:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

            openings.append((game, board.fen()))
                
            game = chess.pgn.read_game(pgn)

    shuffle(openings)

    for i in range(n):
        yield openings[i]
        

class Evaluation:
    def __init__(self, model_1_path: str, model_2_path: str):
        self.model_1 = model_1_path
        self.model_2 = model_2_path
        self.agent_1 = Agent(local_preds=True, model_path=model_1_path)
        self.agent_2 = Agent(local_preds=True, model_path=model_2_path)
        self.score = {
                         "model_1": 0,
                         "model_2": 0,
                         "draws": 0
                     }
                     

    def evaluate(self, n: int = config.EVAL_GAMES, pgn: str = f"{config.PGN}Openings_sf.pgn") -> dict:
        """
        For n openings, let the two models play each other and keep a score
        """
        
        # agent_1 = Agent(local_predictions=True, model_path=self.model_1)
        # agent_2 = Agent(local_predictions=True, model_path=self.model_2)
        i = 0
        for opn, fen in get_openings(n=n, pgn_file=pgn):
            print(f"Playing opening {i + 1}")
            print(f"White: {opn.headers['White']}")
            print(f"Black: {opn.headers['Black']}")
            i += 1
            env = ChessEnv(fen)
            
            # play deterministally
            game = Game(env, self.agent_1, self.agent_2)
            result = game.game(stochastic=False, save=False)
            self.update_score(result, True)
            print(f"Result: {result}")
            print("Final board position: ")
            print(game.env.board)
            
            # turn around the colors
            game = Game(env, self.agent_2, self.agent_1)
            result = game.game(stochastic=False, save=False)
            self.update_score(result, False)
            print(f"Result: {result}")
            print("Final board position: ")
            print(game.env.board)
    
        print(f"Played {2*i} games.")
        print(f"Model 1 won {self.score['model_1']} times.")
        print(f"Model 2 won {self.score['model_2']} times.")
        print(f"Total {self.score['draws']} draws")
  
        return self.score


    def update_score(self, result, agent_1):
        if result == 0: 
            self.score["draws"] += 1
        elif result > 0:
            if agent_1: 
                self.score["model_1"] += 1
            else:
                self.score["model_2"] += 1
        else: 
            if agent_1:
                self.score["model_2"] += 1
            else:
                self.score["model_1"] += 1
            

    def reset_score(self):
        self.score = {
                         "model_1": 0,
                         "model_2": 0,
                         "draws": 0
                     }
        
        
        
if __name__ == "__main__":
    eval = Evaluation(None, None) # replace with model paths
    eval.evaluate(config.EVAL_GAMES)