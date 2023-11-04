# ### This is the Chess Environment that will interact with the Agent

# ### TODO: make this as c as possible

# import numpy as np
# import chess
# import torch
# from copy import deepcopy

# btoi = lambda x: (1 if x else -1) 
# strenc = np.array(['r', 'n', 'b', 'q', 'k','p','P','R', 'N', 'B', 'Q', 'K']).reshape(1,12,1,1)

# ### TODO:docs
# class ChessEnv:
#     def __init__(self,board:str|list,batch_size:int,board_size:int=8,torch_device:str='cuda'):
#         self.batch_size = batch_size
#         self.board_size = board_size
#         self.torch_device = torch_device
#         self.num_piecetype = 12
        
#         self.__init_board_frm_str(board)
        
        
    
#     ### Interface functions ###    
#     def get_embedding(self)->torch.Tensor:
#         self.__update_embedding()
#         ## expand out (_,7) tensor to (_,7,8,8) tensor
#         self.board_states_embedding = self.board_states.unsqueeze(2).unsqueeze(3).repeat(1,1,self.board_size,self.board_size)
#         return torch.cat([self.board_embedding,self.board_states_embedding],dim=1)
        
    
#     ### Functions to convert between different representations of the board ###
#     def __init_board_frm_str(self,board:str|list)->None:
#         if type(board) == str:
#             self.board_init = (chess.Board(board),)
#         elif type(board) == list:
#             self.board_init = [chess.Board(b) for b in board]
#         self.board = self.board_init.deepcopy()
        
#         ### this will be changed !!
#         self.movenum = 0
#         ### load the initial embedding ### -> assuming initially history is repeated rather than empty
#         self.board_embedding = self.__board_to_tensor(self.board).repeat(1,self.board_size,1,1)
        
#         self.reps = torch.zeros((self.batch_size,1),device=self.torch_device)
        
        
        
#     ## remember devices
    
#     ## (turn, 4 castling rights,movenum)
#     def __get_board_states_single(self,board:chess.Board)->None:
#         return torch.tensor([[btoi(board.turn),btoi(board.has_kingside_castling_rights(chess.WHITE)),btoi(board.has_kingside_castling_rights(chess.BLACK)),btoi(board.has_queenside_castling_rights(chess.WHITE)),btoi(board.has_queenside_castling_rights(chess.BLACK)),self.movenum]],device=self.torch_device)
#     ## add reps too
#     def __get_board_states(self)->torch.Tensor:
#         sixvars =  torch.cat([self.__get_board_states_single(b) for b in self.board],dim=0)
#         return torch.cat((sixvars,self.reps),dim=1)
#     # f me #
#     def __board_to_tensor(self,boards:list|tuple)->torch.Tensor:
#         arr = (np.array([b.__str__().split() for b in boards]).reshape(-1,1,self.board_size,self.board_size)==strenc)*1
#         return torch.Tensor(arr, device=self.torch_device)
        
#     def __update_embedding(self)->None: ### maybe this works correctly
#         self.board_embedding = torch.cat([self.board_embedding[:,self.num_piecetype:,:,:],self.__board_to_tensor(self.board)],dim=0)
#         self.reps = (self.reps + 1)*torch.all(torch.all(torch.all(self.board_embedding[:,-2*self.num_piecetype:-self.num_piecetype,:,:]==self.board_embedding[:,-self.num_piecetype:,:,:],dim=3),dim=2),dim=1,keepdim=True)
#         self.board_states = self.__get_board_states()
#         return 
    
#     ## convert moves to mask
#     def __moves_to_mask(self,moves)->torch.Tensor:
#         pass
    
#     ## taking the one-hot encoding of the move chosen and updates the board based on it
#     def __make_move(self,movetensor)->list:
#         pass
    
    
#---------------------------- A simpler implementation for now ---------------

import config
from re import A
import chess
from chess import Move
import numpy as np

import time
import logging

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

def estimate_winner(board: chess.Board) -> int:
    """
    Estimate the winner of the current node.
    Pawn = 1, Bishop = 3, Rook = 5, Queen = 9
    Positive score = white wins, negative score = black wins
    """
    score = 0

    for piece in board.piece_map().values():
        if piece.color == chess.WHITE:
            score += piece_scores[piece.piece_type]
        else:
            score -= piece_scores[piece.piece_type]
    if np.abs(score) > 5:
        if score > 0:
            return 0.25
        else:
            return -0.25
    else:
        return 0

class ChessEnv:
    def __init__(self, fen:str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()
    
    def reset(self):
        self.board = chess.Board(self.fen)
        
    def action(self, action: Move) -> chess.Board:
        '''Perform the given move on the board and return an updated board'''
        self.board.push(action)
        return self.board

    def __str__(self):
        return str(self.board)