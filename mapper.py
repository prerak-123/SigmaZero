from enum import IntEnum
from typing import Tuple
from chess import PieceType
import numpy as np
import config

class QueenDirection(IntEnum):
    # eight directions
    NORTHWEST = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7


class KnightMove(IntEnum):
    # eight possible knight moves
    NORTH_LEFT = 8  # diff == -15
    NORTH_RIGHT = 9  # diff == -17
    EAST_UP = 10  # diff == -6
    EAST_DOWN = 11  # diff == 10
    SOUTH_RIGHT = 12  # diff == 15
    SOUTH_LEFT = 13  # diff == 17
    WEST_DOWN = 14  # diff == 6
    WEST_UP = 15  # diff == -10


class UnderPromotion(IntEnum):
    KNIGHT = 16
    BISHOP = 17
    ROOK = 18


    """
    The mapper is a dictionary of moves.

    * the index is the type of move
    * the value is the plane's index, or an array of plane indices (for distance)
    """
    # knight moves from north_left to west_up (clockwise)

knight_mappings = [-15, -17, -6, 10, 15, 17, 6, -10]

def get_index(piece_type: PieceType, direction: IntEnum, distance: int = 1) -> int:
    if piece_type == PieceType.KNIGHT:
        return 56 + KnightMove(direction + 8).value
    else:
        return QueenDirection(direction) * 8 + distance

def get_underpromotion_move(piece_type: PieceType, from_square: int, to_square: int) -> Tuple[UnderPromotion, int]:
    piece_type = UnderPromotion(piece_type - 2 + 16)
    diff = from_square - to_square
    if to_square < 8:
        # black promotes (1st rank)
        direction = diff - 8
    elif to_square > 55:
        # white promotes (8th rank)
        direction = diff + 8
    return (piece_type, direction)

def get_knight_move(from_square: int, to_square: int) -> KnightMove:
    # print("in knight move")
    return KnightMove(knight_mappings.index(from_square - to_square ) + 8)

def get_queenlike_move(from_square: int, to_square: int) -> Tuple[QueenDirection, int]:
    diff = from_square - to_square
    if diff % 8 == 0:
        # north and south
        if diff > 0:
            direction = QueenDirection.SOUTH
        else:
            direction = QueenDirection.NORTH
        distance = int(diff / 8)
    elif diff % 9 == 0:
        # southwest and northeast
        if diff > 0:
            direction = QueenDirection.SOUTHWEST
        else:
            direction = QueenDirection.NORTHEAST
        distance = np.abs(int(diff / 8)).item()
    elif from_square // 8 == to_square // 8:
        # east and west
        if diff > 0:
            direction = QueenDirection.WEST
        else:
            direction = QueenDirection.EAST
        distance = np.abs(diff).item()
    elif diff % 7 == 0:
        if diff > 0:
            direction = QueenDirection.SOUTHEAST
        else:
            direction = QueenDirection.NORTHWEST
        distance = (np.abs(int(diff / 8)) + 1).item()
    else:
        raise Exception("Invalid queen-like move")
    
    # print(int(direction))
    # print(type(int(direction)), type(distance))
    # print(int(direction), distance)


    return int(direction), distance

mapper = {
    # queens
    QueenDirection.NORTHWEST: [0, 1, 2, 3, 4, 5, 6],
    QueenDirection.NORTH: [7, 8, 9, 10, 11, 12, 13],
    QueenDirection.NORTHEAST: [14, 15, 16, 17, 18, 19, 20],
    QueenDirection.EAST: [21, 22, 23, 24, 25, 26, 27],
    QueenDirection.SOUTHEAST: [28, 29, 30, 31, 32, 33, 34],
    QueenDirection.SOUTH: [35, 36, 37, 38, 39, 40, 41],
    QueenDirection.SOUTHWEST: [42, 43, 44, 45, 46, 47, 48],
    QueenDirection.WEST: [49, 50, 51, 52, 53, 54, 55],
    # knights
    KnightMove.NORTH_LEFT: 56,
    KnightMove.NORTH_RIGHT: 57,
    KnightMove.EAST_UP: 58,
    KnightMove.EAST_DOWN: 59,
    KnightMove.SOUTH_RIGHT: 60,
    KnightMove.SOUTH_LEFT: 61,
    KnightMove.WEST_DOWN: 62,
    KnightMove.WEST_UP: 63,
    # underpromotions
    UnderPromotion.KNIGHT: [64, 65, 66],
    UnderPromotion.BISHOP: [67, 68, 69],
    UnderPromotion.ROOK: [70, 71, 72]
}

def upper_confidence_bound(self, noise: float, W, N) -> float:
        exploration_rate = math.log((1 + self.input_node.N + config.C_base) / config.C_base) + config.C_init
        ucb = exploration_rate * (self.P * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.input_node.turn == chess.WHITE:
            return W / (N + 1) + ucb 
        else:
            return -(self.W / (N + 1)) + ucb