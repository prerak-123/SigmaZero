from agent import Agent
from chessEnv import ChessEnv
from game import Game

class Evaluation:
    def __init__(self, model_1_path: str, model_2_path: str):
        self.model_1 = model_1_path
        self.model_2 = model_2_path

    def evaluate(self, n: int) -> dict:
        """
        For 2n games, let the two models play each other and keep a score
        """
        score = {
            "model_1": 0,
            "model_2": 0,
            "draws": 0
        }
        agent_1 = Agent(local_predictions=True, model_path=self.model_1)
        agent_2 = Agent(local_predictions=True, model_path=self.model_2)
        for i in range(n):
            print(f"Playing {i + 1}/n game")
            game = Game(ChessEnv(), agent_1, agent_2)
            # play deterministally
            result = game.play_one_game(stochastic=False)
            if result == 0: score["draws"] += 1
            elif result == 1: score["model_1"] += 1
            else: score["model_2"] += 1
            # turn around the colors
            game = Game(ChessEnv(), agent_2, agent_1)
            result = game.play_one_game(stochastic=False)
            if result == 0: score["draws"] += 1
            elif result == 1: score["model_2"] += 1
            else: score["model_1"] += 1
    
        print(f"Played {2*n} games.\nModel 1 won {score['model_1']} times.\nModel 2 won {score['model_2']} times.\nTotal {score['draws']} draws")
  
        return score


