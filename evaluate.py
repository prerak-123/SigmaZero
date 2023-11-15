from agent import Agent
from chessEnv import ChessEnv
from game import Game
import chess.pgn

def get_openings(n: int = -1, pgn_file: str = "Openings_sf.pgn"):
    pgn = open(pgn_file)
    
    game = chess.pgn.read_game(pgn)
    while game is not None:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            
        n -= 1
        if n:
            yield game, board.fen()
        else:
            return game, board.fen()
        
        game = chess.pgn.read_game(pgn)

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

    def evaluate(self, n: int = -1, pgn: str = "Openings_sf.pgn") -> dict:
        """
        For n openings, let the two models play each other and keep a score
        """
        
        # agent_1 = Agent(local_predictions=True, model_path=self.model_1)
        # agent_2 = Agent(local_predictions=True, model_path=self.model_2)
        i = 1
        for opn, fen in get_openings(n=n, pgn_file=pgn):
            print(f"Playing opening {i}")
            print(f"White: {opn.headers['White']}")
            print(f"Black: {opn.headers['Black']}")
            i += 1
            env = ChessEnv(fen)
            
            # play deterministally
            game = Game(env, self.agent_1, self.agent_2)
            result = game.game(stochastic=False, save=False)
            self.update_score(result)
            print(f"Result: {result}")
            print("Final board position: ")
            print(game.env.board)
            
            # turn around the colors
            game = Game(env, self.agent_2, self.agent_1)
            result = game.game(stochastic=False, save=False)
            self.update_score(result)
            print(f"Result: {result}")
            print("Final board position: ")
            print(game.env.board)
    
        print(f"Played {2*i} games.")
        print(f"Model 1 won {self.score['model_1']} times.")
        print(f"Model 2 won {self.score['model_2']} times.")
        print(f"Total {self.score['draws']} draws")
  
        return self.score

    def update_score(self, result):
        if result == 0: 
            self.score["draws"] += 1
        elif result == 1: 
            self.score["model_1"] += 1
        else: 
            self.score["model_2"] += 1
            
    def reset_score(self):
        self.score = {
                         "model_1": 0,
                         "model_2": 0,
                         "draws": 0
                     }
        
        
        
if __name__ == "__main__":
    eval = Evaluation(None, None)
    eval.evaluate()