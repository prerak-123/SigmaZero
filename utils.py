import chess
from chess import Move
import numpy as np
import mapper

def moves_to_output_vector(moves: dict, board: chess.Board) -> np.ndarray:
    """
    Convert a dictionary of moves to a vector of probabilities
    """
    vector = np.zeros((73, 8, 8), dtype=np.float32)
    for move in moves:
        plane_index, row, col = move_to_plane_index(move, board)
        vector[plane_index, row, col] = moves[move]

    return vector
    
def move_to_plane_index(move: str, board: chess.Board):
    """"
    Convert a move to a plane index and the row and column on the board
    """
    move: Move = Move.from_uci(move)
    # get start and end position
    from_square = move.from_square
    to_square = move.to_square
    # get piece
    piece: chess.Piece = board.piece_at(from_square)

    if piece is None:
            raise Exception(f"No piece at {from_square}")

    plane_index: int = None

    if move.promotion and move.promotion != chess.QUEEN:
        piece_type, direction = mapper.get_underpromotion_move(
            move.promotion, from_square, to_square
        )
        plane_index = mapper.mapper[piece_type][1 - direction]
    else:
        if piece.piece_type == chess.KNIGHT:
            # get direction
                direction = mapper.get_knight_move(from_square, to_square)
                plane_index = mapper.mapper[direction]
        else:
            # get direction of queen-type move
            direction, distance = mapper.get_queenlike_move(
                from_square, to_square)
            plane_index = mapper.mapper[direction][np.abs(distance)-1]
    row = from_square % 8
    col = 7 - (from_square // 8)
    return (plane_index, row, col)

