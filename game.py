import chess
from config import *
import copy
import numpy as np

class Game:
    def __init__(self, board: chess.Board, white: Agent, black: Agent):
        """
        Class used to play games for both training and testing.

        Args:
            board (ChessEnv): Starting board position
            white (Agent): White engine
            black (Agent): Black engine
        """
        self.board = board
        self.white = white
        self.black = black
        
        
    def result(self):
        """
        Gives the result of the game if it has finished.

        Returns:
            An integer representing white's result.
            
            1: White won
            0: Black won
            1/2: Draw
            -1: No result
        """
        
        res = self.env.board.result()
        if res == "1-0":
            return 1
        elif res == "0-1":
            return 0
        elif res == "1/2-1/2":
            return 1/2
        else:
            return -1
        
        
    # A good way to estimate result of incomplete game
    def estimate_result():
        pass
        
        
    def game(self, training: bool = True):
        """
        Play a game from the starting position.

        Args:
            training (bool, optional): Is this a training game? Defaults to True.
        """
        self.game_history = [[None, None] for _ in range(PREVIOUS_MOVES)]
        self.game_board = copy.deepcopy(self.board)
        
        while self.result() == -1:
            prev_move = self.move(training)
            
            if self.game_board.is_seventyfive_moves():
                break
            
        winner = self.result()
        if winner == -1:
            winner = self.estimate_result()
            
        print("Final game position")
        print(self.game_board)
            
        return winner
        
    
    def move(self, training: bool = True):
        """
        Plays one move after running a mcts simulation from current position.

        Returns:
            The move played by the agent.
        """
        player = self.white if self.game_board.turn else self.black
        
        # Create a mcts tree from the current position for player
        # Run the mcts algorithm multiple times
        # Get the evaluations for different moves(edges)
        moves = player.mcts.moves
        prob = [m.visit/player.mcts.total_sim for m in moves]
        
        if training:
            next_move = np.random.choice(moves, p=prob)
        else:
            next_move = moves[np.argmax(prob)]
        
        if self.game_board.turn:
            self.game_history = self.game_history[1:].append([next_move, None])
        else:
            self.game_history[-1][1] = next_move
            
        # Play the move(edge)
        self.game_board = play_move(next_move)
        
        return next_move