### This is the Chess Environment that will interact with the Agent and the MCTS
#---------------------------- A simpler implementation for now ---------------

import config
from re import A
import chess
from chess import Move
import numpy as np

import time
import logging

from stockfish import Stockfish
import os

def state_to_input(fen: str):

    board = chess.Board(fen)
    
    state = []

    # 1. is it white's turn? (1x8x8)
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.turn == chess.WHITE else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))
    
    # 2. Castling Rights (4x8x8)
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))

    # 3. repitition counter
    state.append(np.ones((config.BOARD_SIZE, config.BOARD_SIZE)) if board.can_claim_fifty_moves() else np.zeros((config.BOARD_SIZE, config.BOARD_SIZE)))

    # 4. Pieces Position
    for color in chess.COLORS:
        # 4. player 1's pieces (6x8x8)
        # 5. player 2's pieces (6x8x8)
        for piece_type in chess.PIECE_TYPES:
            # 6 arrays of 8x8 booleans
            piece_pos = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE))
            for index in list(board.pieces(piece_type, color)):
                # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                piece_pos[7 - int(index/8)][index % 8] = 1
            state.append(piece_pos)
    
    en_passant = np.zeros((8, 8))
    if board.has_legal_en_passant():
        en_passant[7 - int(board.ep_square/8)][board.ep_square % 8] = True
    
    state.append(en_passant)

    state = np.asarray(state, dtype=bool)
    del board
    return state

piece_scores = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

### ensure stockfish is installed and the path is correct
stockfish = Stockfish(os.path.expanduser(config.STOCKFISH))

def estimate_position(board: chess.Board) -> int:
    """
    Currently uses stockfish's evaluation function, but can be modified to use a simpler or more complicated heuristic based on need.
    """

    try:
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()
    except:
        stockfish = Stockfish(os.path.expanduser(config.STOCKFISH))
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()

    if evaluation['type'] == 'cp':
        return float(np.tanh(evaluation['value'] / 100))
    elif evaluation['type'] == 'mate':
        return float(np.sign(evaluation['value']))
    else:
        print('WRONG EVALUATION TYPE BY STOCKFISH')
        assert(False)

class ChessEnv:
    def __init__(self, fen:str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()
    
    def reset(self):
        self.board = chess.Board(self.fen)
        
    def step(self, action: Move) -> chess.Board:
        '''Perform the given move on the board and return an updated board'''
        self.board.push(action)
        return self.board

    def __str__(self):
        return str(self.board)
