from neural_network import AgentNetwork
import chess
from transform import board_to_input
import config
import matplotlib.pyplot as plt
import torch
from agent import Agent
from utils import moves_to_output_vector


device = 'cuda' if torch.cuda.is_available() else 'cpu'

nn = AgentNetwork(input_channels=config.IN_CHANNELS, num_hidden_blocks=config.NUM_BLOCKS).to(device)

STR = "4r2k/8/3PP3/P1P5/p3N3/p4P2/2ppp3/K2n1B2 w - - 0 1"
board = chess.Board(STR)

print("Board:")
print(board)

state = board_to_input(board).to(device).unsqueeze(dim=0)

_, action = nn(state)

action = action.reshape((-1, 8, 8)).cpu().detach().numpy()

for indx, img in enumerate(action):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Random Model-Plane Index {indx}")
    plt.savefig(f"action_figures/random-{indx}.png")
    plt.clf()
    
agent = Agent(local_preds=True, state=STR)
agent.run_simulations(config.SIMULATIONS_PER_MOVE)
moves = []
moves = agent.mcts.get_all_edges(moves)

sum_move_visits = 0
for e in moves:
    sum_move_visits += agent.mcts.get_edge_N(e)
        
search_probabilities = {
agent.mcts.get_edge_action(e).uci(): agent.mcts.get_edge_N(e) / sum_move_visits for e in moves}

policy = moves_to_output_vector(search_probabilities, board)
for indx, img in enumerate(policy):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Random Model - MCTS Policy - Plane Index {indx}")
    plt.savefig(f"action_figures/random-mcts-{indx}.png")
    plt.clf()


nn.load_state_dict(torch.load(f"{config.BEST_MODEL}best-model.pth", map_location="cpu"))

_, action = nn(state)

action = action.reshape((-1, 8, 8)).cpu().detach().numpy()

for indx, img in enumerate(action):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Trained Model-Plane Index {indx}")
    plt.savefig(f"action_figures/trained-{indx}.png")
    plt.clf()

agent = Agent(local_preds=True, model_path=f"{config.BEST_MODEL}best-model.pth", state=STR)
agent.run_simulations(config.SIMULATIONS_PER_MOVE)
moves = []
moves = agent.mcts.get_all_edges(moves)

sum_move_visits = 0
for e in moves:
    sum_move_visits += agent.mcts.get_edge_N(e)
        
search_probabilities = {
agent.mcts.get_edge_action(e).uci(): agent.mcts.get_edge_N(e) / sum_move_visits for e in moves}

policy = moves_to_output_vector(search_probabilities, board)

for indx, img in enumerate(policy):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Trained Model - MCTS Policy - Plane Index {indx}")
    plt.savefig(f"action_figures/trained-mcts-{indx}.png")
    plt.clf()