from neural_network import AgentNetwork
import chess
from transform import board_to_input
import config
import matplotlib.pyplot as plt
import torch

nn = AgentNetwork(input_channels=config.IN_CHANNELS, num_hidden_blocks=config.NUM_BLOCKS)

STR = "4r2k/8/3PP3/P1P5/p3N3/p4P2/2ppp3/K2n1B2 w - - 0 1"
board = chess.Board(STR)

print("Board:")
print(board)

state = board_to_input(board).unsqueeze(dim=0)


_, action = nn(state)

action = action.reshape((-1, 8, 8)).detach().numpy()

for indx, img in enumerate(action):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Random Model-Plane Index {indx}")
    plt.savefig(f"action_figures/random-{indx}.png")
    plt.clf()

nn.load_state_dict(torch.load(f"{config.BEST_MODEL}best-model.pth", map_location="cpu"))

_, action = nn(state)

action = action.reshape((-1, 8, 8)).detach().numpy()

for indx, img in enumerate(action):
    plt.imshow(img, cmap='grey', origin='lower')
    plt.colorbar()
    plt.title(f"Trained Model-Plane Index {indx}")
    plt.savefig(f"action_figures/trained-{indx}.png")
    plt.clf()