BOARD_SIZE = 8
MAX_MOVES = 75
PREVIOUS_MOVES = 8
SIMULATIONS_PER_MOVE = 800
EVAL_GAMES = 4

#-----------Model Parameters------------------------
IN_CHANNELS = 19
NUM_BLOCKS = 19

# --------- Training Parameters --------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.2 #Decay to 0.02, 0.002....?
TRAIN_STEPS = 10000

#-----------Directories---------------------------
IMAGES = "./images/"
MODEL = "./models/"
MEMORY = "./memory/"
PUZZLE = "./puzzles/"
BEST_MODEL = "./best_model/"