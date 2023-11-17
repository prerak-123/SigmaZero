BOARD_SIZE = 8
MAX_MOVES = 50
PREVIOUS_MOVES = 8
SIMULATIONS_PER_MOVE = 800
EVAL_GAMES = 3

PUZZLE_PROB = 0.2

#-----------Model Parameters------------------------
IN_CHANNELS = 19
NUM_BLOCKS = 19

# --------- Training Parameters --------------------
BATCH_SIZE = 256
LEARNING_RATE = 0.002 #Decay to 0.02, 0.002....?
TRAIN_STEPS = 40000

#-----------Directories---------------------------
IMAGES = "./images/"
MODEL = "./models/"
MEMORY = "./memory/"
PUZZLE = "./puzzles/"
BEST_MODEL = "./best_model/"

#----------Executable Locations--------------------
STOCKFISH = "~/stockfish/stockfish-ubuntu-x86-64-modern"