# Play a human vs agent game
from agent import Agent
from chessEnv import ChessEnv
import config
import numpy as np
import config
from datetime import datetime
import time

from CPP_backend import MCTS
import chess

import pygame
import math

# display size
X = 800
Y = 800
# create screen
scrn = pygame.display.set_mode((X, Y))
pygame.init()
# colors
WHITE = (255, 255, 255)
LGREY = (238,238,210)
DGREY = (118,150,86)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)
# create board object
b = chess.Board()
# load images
pieces = {
    'p': pygame.transform.scale(pygame.image.load('./gui/images/b_pawn.png'), (100, 100)),
    'n': pygame.transform.scale(pygame.image.load('./gui/images/b_knight.png'), (100, 100)),
    'b': pygame.transform.scale(pygame.image.load('./gui/images/b_bishop.png'), (100, 100)),
    'r': pygame.transform.scale(pygame.image.load('./gui/images/b_rook.png'), (100, 100)),
    'q': pygame.transform.scale(pygame.image.load('./gui/images/b_queen.png'), (100, 100)),
    'k': pygame.transform.scale(pygame.image.load('./gui/images/b_king.png'), (100, 100)),
    'P': pygame.transform.scale(pygame.image.load('./gui/images/w_pawn.png'), (100, 100)),
    'N': pygame.transform.scale(pygame.image.load('./gui/images/w_knight.png'), (100, 100)),
    'B': pygame.transform.scale(pygame.image.load('./gui/images/w_bishop.png'), (100, 100)),
    'R': pygame.transform.scale(pygame.image.load('./gui/images/w_rook.png'), (100, 100)),
    'Q': pygame.transform.scale(pygame.image.load('./gui/images/w_queen.png'), (100, 100)),
    'K': pygame.transform.scale(pygame.image.load('./gui/images/w_king.png'), (100, 100)),
}

def get_winner(result: str) -> int:
    return 1 if result == "1-0" else - 1 if result == "0-1" else 0
      
def chess_avh(BOARD):
    '''
    agent vs human game
    '''
    pygame.event.clear()
    # variable to be used later
    index_moves = []
    all_moves = list(BOARD.legal_moves)
    move = None

    while move == None:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.MOUSEBUTTONDOWN:

                scrn.fill((222, 184, 136))

                # get pos of mouse
                pos = pygame.mouse.get_pos()

                # find which square was clicked and index of it
                square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                index = (7 - square[1]) * 8 + (square[0])

                # if we have already highlighted moves and are making a move
                if index in index_moves:
                    move = moves[index_moves.index(index)]
                    index = None
                    index_moves = []
                # highlight possible moves
                else:
                    piece = BOARD.piece_at(index)

                    if piece is None:
                        pass
                    else:
                        moves = []
                        for human_mvt in all_moves:
                            if human_mvt.from_square == index:
                                moves.append(human_mvt)

                                t = human_mvt.to_square

                                TX1 = 100 * (t % 8)
                                TY1 = 100 * (7 - t // 8)

                                pygame.draw.rect(scrn, GREY, pygame.Rect(TX1, TY1, 100, 100), 5)

                        index_moves = [a.to_square for a in moves]

        update_board(scrn, BOARD)

    if BOARD.outcome() is not None:
        print(BOARD.outcome())
        print(BOARD)

    return move

def update_board(scrn, board):
    # Function to update the chessboard display
    # scrn.fill((222, 184, 136))

    for i in range(9):
            for j in range(8):
                if (i + j) % 2 == 0:
                    pygame.draw.rect(scrn, LGREY, pygame.Rect(i * 100, j * 100, 100, 100))
                else:
                    pygame.draw.rect(scrn, DGREY, pygame.Rect(i * 100, j * 100, 100, 100))

    for i in range(1,8):
        # i = i + 1
        pygame.draw.line(scrn, WHITE, (0, i * 100), (800, i * 100))
        pygame.draw.line(scrn, WHITE, (i * 100, 0), (i * 100, 800))

    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            scrn.blit(pieces[str(piece)], ((i % 8) * 100, 700 - (i // 8) * 100))

    pygame.display.flip()

class Game:
    def __init__(self, env: ChessEnv, agent: Agent):
        """
        Class used to play games for both training and testing.

        Args:
            board (ChessEnv): Starting board position
            white (Agent): White engine
            black (Agent): Black engine
        """
        self.env = env
        self.agent = agent
        
        
        self.reset()
        
    def reset(self):
        self.env.reset()
    
    def game(self):
        """
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        """
        pygame.init()
        # create screen
        scrn = pygame.display.set_mode((X, Y))
        scrn.fill((222, 184, 136))
        pygame.display.set_caption('Chess')

        update_board(scrn, self.env.board)

        # reset everything
        self.reset()
        # add a new memory entry
        # counter to check amount of moves played. if above limit, estimate winner
        while not self.env.board.is_game_over():
            # play one move (previous move is used for updating the MCTS tree)
            human_mvt = chess_avh(self.env.board)
            self.env.step(human_mvt)

            update_board(scrn, self.env.board)

            self.play_move()

            update_board(scrn, self.env.board)
        
        time.sleep(5)
        pygame.quit()
        
        winner = get_winner(self.env.board.result())
        # save game result to memory for all games
        
        return winner
        
    
    def play_move(self) -> None:
        """
        Plays one move after running a mcts simulation from current position.

        Returns:
            The move played by the agent.
        """
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
        node found after playing the previous moves in the current tree.
        """
        # whose turn is it
        current_player = self.agent

        current_player.mcts = MCTS(current_player, self.env.board.fen(), False)

        t1 = time.time()
        current_player.run_simulations(n=config.SIMULATIONS_PER_MOVE)
        t2 = time.time()
        print(f"Time taken for {config.SIMULATIONS_PER_MOVE} simulations: {t2 - t1}")


        moves = []
        moves = current_player.mcts.get_all_edges(moves)

        sum_move_visits = current_player.mcts.get_sum_N()
        probs = [current_player.mcts.get_edge_N(e) / sum_move_visits for e in moves]
        moves = np.array(moves)
        
        best_move = np.int64(moves[np.argmax(probs)])

        # print("best move", current_player.mcts.get_edge_uci(best_move.item()))
        self.env.step(current_player.mcts.get_edge_action(best_move.item()))        


if __name__ == '__main__':
    env = ChessEnv()
    black=Agent(local_preds=True)
    g = Game(env, black)
    g.game()
