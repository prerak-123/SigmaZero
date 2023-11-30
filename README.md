# SigmaZero
Implementation of AlphaZero as part of the course project in CS337.

The link to the report describing the implementations and results obtained can be found [here](https://www.cse.iitb.ac.in/~kartikn/docs/SigmaZero.pdf)

# Results

## Model Weights

[Link](https://drive.google.com/file/d/12QiXUTJTqZ05LSDosasyz2efBJ46E7CS/view?usp=sharing) to the weights of the final model.

## Plots

All the plots obtained during training can be found [here](https://drive.google.com/drive/folders/1j1uye2L0v_v4zI3UOE_RFkkT7dXK5Cai?usp=sharing)

## Visualizing action policy

The figures obtained to visualize the action policy tensor can be found [here](https://drive.google.com/drive/folders/1VcsbNjf1xEZX4O0cB90V-mjies__e_qC?usp=sharing)

## Video of a game against SigmaZero

Video of a game played against the trained model on our GUI can be found [here](https://drive.google.com/drive/u/1/folders/18x4usppypEMzvOBg392J9WI2OyPgjE6_)

# Instructions to run the code
## Instructions to Compile C++ backend

- Install Boost library from [here](https://www.boost.org/doc/libs/1_46_1/more/getting_started/unix-variants.html)
- To compile C++ backend into a shared object file use : `g++ -O3 CPP_backend.cpp -shared -fpic -Wno-undef -I BOOST_INSTALL_PATH -I /usr/include/python3.x/ -L BOOST_INSTALL_PATH/stage/lib/  -lboost_python3x -lboost_system -o CPP_backend.so`
- Replace `x` with appropriate python version number installed in your machine. Python 3.10 is recommended.

## Stockfish Engine

The evaluation function requires stockfish engine, which can be downloaded from [this](https://stockfishchess.org/download/) link.

Once built, the path of the executable needs to then be stored in the variable ``STOCKFISH`` in ``config.py``

## Puzzles

The collection of experiences requires a puzzles dataset. The dataset used by us is the one provided by Lichess. It can be downloaded from [this](https://database.lichess.org/#puzzles) link. 

The csv needs to be stored inside ``puzzles`` directory, named as ``puzzles.csv``.

## Conda Environment

The dependencies for the conda environment can be found in ``sigma.yaml``, and the environment can be set up using the following command ``conda env create -f sigma.yaml``

## Directory Structure

Make the following directories in the current directory:

- best_model
- memory
- images

## Run the code

To run ``collect.py``, use the command ``python3 collect.py <start_id>``, where start_id corresponds to the index puzzle from which you want to start training

To run ``train.py`` use the command ``python3 train.py``

You can change all the hyper parameters by changing their values in ``config.py``

# Files Structure

The open source implementation of AlphaZero by Tuur Vanhoutte was used as the main reference while implementing SigmaZero, and can be found [here](https://github.com/zjeffer/chess-deep-rl)

The various files and their functionalities are as follows:

- __agent.py__

Contains the implementation of class ``Agent`` which maintains an instance of the Neural Network and an instance of a MCTS tree. Has the functionalities of performing simulations on the MCTS tree and making the forward pass in the neural network.

There was an initial idea to use a server client based approach to parallelize collection of experiences, however it is kept on hold currently, and replaced by a simpler multi processing approach, and hence attempting to try server prediction will raise a NotImplementedError.

- __chessEnv.py__

Contains methods to compute the state tensor from a chess board and estimate position.

Also contains the implementation of the class ``ChessEnv`` which serves as the environment for the RL algorithm, storing the current position of board as state and has a function to perform a step (make a move).

- __chessgui.py__

A simple GUI implementation to play against the agent. Requires weights to be present in the location ``best_model\best-model.pth``.

- __collect.py__

The script which collects experiences using self-play. The proportion of puzzles and full games can be tuned in ``config.py``. 

The script creates multiple processes, each parallely collecting experiences. The number of processes can be changed by changing the value of variable ``N`` in the file.

- __CPP_backend.cpp__

An implementation of the MCTS algorithm, fully written in C++ to gain a speed-up from a Python implementation. The interface for Python is created using ``boost``. The main functionalities exposed are: Performing simulations on MCTS algorithm and getting the MCTS policy for the root node.

The C++ code handles memory allocation and iterative aspects of MCTS algorithm, while using python functionalities to communicate with the Neural Network of the Agent and perform certain mathematical operations like ``log`` and ``sqrt``

- __evaluate.py__

Contains an implementation of the class ``Evaluation`` which evaluates two different models on different openings and returns a score of the two models. 

The script randomly samples openings from a set of several openings as the initial position and lets both the agents play once as white and once as black.

- __game.py__

Contains an implementation of the class ``Game`` which has the functionality of playing games of chess, given two agents and an environment. 

The class also has functionalities of maintaining a memory for the game, and also storing the experiences in a directory after the game is over.

- __mapper.py__

Contains implementations for mapping chess moves to corresponding indices in the action tensor, used to convert a policy to a tensor and vice-versa.

- __neural_network.py__

Contains implementation of the class ``AgentNetwork``, which is the neural network used by agent to make predictions. Has functionalities of building the network and performing forward passes.

- __train.py__

The script to optimise a model on the collected experiences. The script collects all the experiences, keeps sampling mini batches and performs backward passes on the loss. 

After each training iteration, it evaluates the newly obtained model against the previous best, and updates the best model if the current model outperforms the previous one.

- __transform.py__

Contains functionalities to convert a board to the corresponding state tensor.

- __util.py__

Contains utility functions to convert policy to action tensor.

- __visualize.py__

The script to visualize the planes of the policy for a trained model and compare it with a randomly initialised model. The FEN string of the board on which you want to perform the visualisation can be set in the ``STR`` variable. 

The script can be run using the command ``python3 visualize.py``
