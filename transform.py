import chess
import torch

def board_to_input(board: chess.Board):
    """Coverts the given board position into the planes 
    required as input for the neural network.
    """
    pieces = []
    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            plane = torch.zeros(1, 8, 8)
            for index in list(board.pieces(piece_type, color)):
                plane[0][7 - int(index/8)][index % 8] = True
                
            pieces.append(plane)
            
    turn = torch.ones(1, 8, 8) if board.turn else torch.zeros(1, 8, 8)
    
    moves = torch.full((1, 8, 8), board.fullmove_number)
    
    wk_castle = torch.ones(1, 8, 8) if board.has_kingside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
    wq_castle = torch.ones(1, 8, 8) if board.has_queenside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
    bk_castle = torch.ones(1, 8, 8) if board.has_kingside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
    bq_castle = torch.ones(1, 8, 8) if board.has_queenside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
    
    claim_draw = torch.ones(1, 8, 8) if board.can_claim_draw() else torch.zeros(1, 8, 8)
    
    return torch.cat((*pieces, turn, moves, wk_castle, wq_castle, bk_castle, bq_castle, claim_draw), dim=0)