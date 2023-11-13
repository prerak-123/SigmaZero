import chess
import torch


def board_to_input(board: chess.Board)->torch.Tensor:
    """
    Coverts the given board position into the planes 
    required as input for the neural network.
    """
    pieces = []
    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            plane = torch.zeros(1, 8, 8)
            for index in list(board.pieces(piece_type, color)):
                plane[0][7 - index//8][index % 8] = True
                
            pieces.append(plane)
            
    turn = torch.ones(1, 8, 8) if board.turn else torch.zeros(1, 8, 8)
    
    moves = torch.full((1, 8, 8), board.fullmove_number)
    
    wk_castle = torch.ones(1, 8, 8) if board.has_kingside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
    wq_castle = torch.ones(1, 8, 8) if board.has_queenside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
    bk_castle = torch.ones(1, 8, 8) if board.has_kingside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
    bq_castle = torch.ones(1, 8, 8) if board.has_queenside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
    
    claim_draw = torch.ones(1, 8, 8) if board.can_claim_draw() else torch.zeros(1, 8, 8)
    
    return torch.cat((*pieces, turn, moves, wk_castle, wq_castle, bk_castle, bq_castle, claim_draw), dim=0)


def move_to_output(move: chess.Move):
    knight_moves = [-15, -17, -6, 10, 15, 17, 6, -10]
    
    diff = move.to_square - move.from_square
    index = 0
    
    if move.promotion == chess.QUEEN or move.promotion is None:
        if diff not in knight_moves:
            if diff%8 == 0:
                index = abs(diff)//8
                
            elif move.to_square//8 == move.from_square//8:
                index = 14 + abs(diff)
                
            elif move.to_square%8 < move.from_square%8:
                index = 28
                dist = diff - (move.to_square%8 - move.from_square%8)
                index += abs(dist)//8
                    
            else:
                index = 42
                dist = diff - (move.to_square%8 - move.from_square%8)
                index += abs(dist)//8
                
            if diff < 0:
                    index += 7
                    
        else:
            index = 56 + knight_moves.index(diff) + 1
    else:
        index = 64 + 3*(move.promotion - 2) + (abs(diff) - 6)
        
    return index-1
