# Parameters for all the files #

BOARD_SIZE = 8
MAX_MOVES = 150
PREVIOUS_MOVES = 8
SIMULATIONS_PER_MOVE = 800
EVAL_GAMES = 3

PUZZLE_PROB = 0.6

#-----------Model Parameters------------------------
IN_CHANNELS = 19
NUM_BLOCKS = 19

# --------- Training Parameters --------------------
BATCH_SIZE = 128
LEARNING_RATE = 0.002 #Decay to 0.02, 0.002....?
WEIGHT_DECAY=1e-4
TRAIN_STEPS = 100000

#-----------Directories---------------------------
IMAGES = "./images/"
MODEL = "./models/"
MEMORY = "./memory/"
PUZZLE = "./puzzles/"
BEST_MODEL = "./best_model/"
PGN = "./pgn/"

#----------Executable Locations--------------------
STOCKFISH = "~/stockfish/stockfish-ubuntu-x86-64-modern"
