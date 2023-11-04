from agent import Agent
from chessEnv import ChessEnv
import config
import numpy as np
import config
from datetime import datetime

def get_winner(result: str) -> int:
    return 1 if result == "1-0" else - 1 if result == "0-1" else 0

class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent):
        """
        Class used to play games for both training and testing.

        Args:
            board (ChessEnv): Starting board position
            white (Agent): White engine
            black (Agent): Black engine
        """
        self.env = env
        self.white = white
        self.black = black
        
        self.memory = []
        
        self.reset()
        
    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn  # True = white, False = black
        
        
    def game(self, stochastic: bool = True):
        """
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        """
        # reset everything
        self.reset()
        # add a new memory entry
        self.memory.append([])
        # counter to check amount of moves played. if above limit, estimate winner
        counter, previous_edges, full_game = 0, (None, None), True
        while not self.env.board.is_game_over():
            # play one move (previous move is used for updating the MCTS tree)
            previous_edges = self.play_move(stochastic=stochastic, previous_moves=previous_edges)

            # end if the game drags on too long
            counter += 1
            if counter > config.MAX_MOVES or self.env.board.is_repetition(3):
                # estimate the winner based on piece values
                winner = ChessEnv.estimate_winner(self.env.board)
                full_game = False
                break
        if full_game:
            # get the winner based on the result of the game
            winner = get_winner(self.env.board.result())
        # save game result to memory for all games
        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        self.save_game(name="game", full_game=full_game)

        return winner
        
    
    def play_move(self, stochastic: bool = True, previous_moves = (None, None), save_moves = True) -> None:
        """
        Plays one move after running a mcts simulation from current position.

        Returns:
            The move played by the agent.
        """
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
        node found after playing the previous moves in the current tree.
        """
        # whose turn is it
        current_player = self.white if self.turn else self.black

        if previous_moves[0] is None or previous_moves[1] is None:
            # create new tree with root node == current board
            current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        else:
            # change the root node to the node after playing the two previous moves
            try:
                node = current_player.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node
                current_player.mcts.root = node
            except AttributeError:
                current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        # play n simulations from the root node
        current_player.run_simulations(n=config.SIMULATIONS_PER_MOVE)

        moves = current_player.mcts.root.edges

        if save_moves:
            self.save_to_memory(self.env.board.fen(), moves)

        sum_move_visits = sum(e.N for e in moves)
        probs = [e.N / sum_move_visits for e in moves]
        
        if stochastic:
            # choose a move based on a probability distribution
            best_move = np.random.choice(moves, p=probs)
        else:
            # choose a move based on the highest N
            best_move = moves[np.argmax(probs)]

        # play the move
        self.env.step(best_move.action)
        
        # switch turn
        self.turn = not self.turn

        # return the previous move and the new move
        return (previous_moves[1], best_move)
    
    def save_to_memory(self, state, moves) -> None:
        """
        Append the current state and move probabilities to the internal memory.
        """
        sum_move_visits = sum(e.N for e in moves)
        # create dictionary of moves and their probabilities
        search_probabilities = {
            e.action.uci(): e.N / sum_move_visits for e in moves}
        # winner gets added after game is over
        self.memory[-1].append((state, search_probabilities, None))

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        # the game id consist of game + datetime
        game_id = f"{name}-Full-{full_game}-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        np.save(f"{config.MEMORY}{game_id}.npy", self.memory[-1])