import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import gym
from gym import spaces
import inspect
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

DEBUG = True

import os
import datetime

global myfig, myax

num_episodes = 2000
num_eval_episodes = 1000
epsilon = 0.1


def enter_function(frame, data=None):
    if DEBUG:
        with open('training_log1.txt', 'a') as f:
            # Redirect stdout to the file
            sys.stdout = f
            # frame = inspect.currentframe()
            function_name = inspect.getframeinfo(frame).function
            print("Entering function:", function_name, "with data = ", data)
            # Restore stdout back to the console
            sys.stdout = sys.__stdout__


def leave_function(frame, data=None):
    if DEBUG:
        with open('training_log1.txt', 'a') as f:
            # Redirect stdout to the file
            sys.stdout = f
            # frame = inspect.currentframe()
            function_name = inspect.getframeinfo(frame).function
            print("Leaving function:", function_name, "with data = ", data)
            # Restore stdout back to the console
            sys.stdout = sys.__stdout__


class SimplicialComplexGameWrapper(gym.Wrapper):
    @property
    def unwrapped(self):
        return self.env

    def __init__(self, env):
        super().__init__(env)
        self.metadata = {'render.modes': []}
        self.num_vertices = env.num_vertices
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.num_vertices, self.num_vertices))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        #enter_function(inspect.currentframe(), obs)

        adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices))
        adjacency_matrix[:obs[2].shape[0], :obs[2].shape[1]] = obs[2]

        # # Preserve the edge labels
        # adjacency_matrix[:obs[2].shape[0], :obs[2].shape[1]] = np.where(obs[2] != 0, obs[2],
        #                                                                 adjacency_matrix[:obs[2].shape[0],
        #                                                                 :obs[2].shape[1]])

        #leave_function(inspect.currentframe(), adjacency_matrix)
        return adjacency_matrix


def enter_vertices():
    vert = input("Number of Vertices: ")
    if not vert.isnumeric():
        print("Invalid input: must be a natural number")
        return enter_vertices()
    else:
        v = int(vert)
        return v


def enter_order(num_vertices):
    ord = input("Order of losing clique: ")
    if not ord.isnumeric():
        print("Invalid input: must be a natural number")
        return enter_order(num_vertices)
    o = int(ord)
    if o > num_vertices:
        print("Invalid input: order must be less than the number of vertices")
        return enter_order(num_vertices)
    return o


def getEdges(distance):
    """
        puts all edges into an array

        Parameters
        ----------
        distance : ndarray
            The pairwise distance matrix between points.

        Returns
        -------
        edges : array
            array containing all pairs of vertices that has a filled edge between them
    """
    edges = []
    for v1 in range(distance.shape[0]):
        for v2 in range(distance.shape[0]):
            if (distance[v1][v2] != 0):
                e = np.array([v1, v2])
                edges.append(e)
    edges = np.array(edges)
    return edges


def create_simplicial_complex(D, r):
    """
    Input: distance matrix and nonnegative radius
    Output: networkx graph
    """

    G = nx.Graph()
    G.add_nodes_from(list(range(len(D))))
    # edge_list = np.argwhere(D <= r)
    edge_list = getEdges(D)
    G.add_edges_from(edge_list)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


class SimplicialComplexGame:
    def __init__(self, num_vertices, order):
        self.num_vertices = num_vertices
        self.order = order
        self.turn = 1
        self.r = 0
        self.outcome = -1
        self.player1 = np.zeros(shape=(num_vertices, num_vertices))
        self.player2 = np.zeros(shape=(num_vertices, num_vertices))
        self.action_space = spaces.Discrete(num_vertices * (num_vertices - 1) // 2)
        print('self.action_space', self.action_space)
        self.observation_space = spaces.Box(low=0, high=2, shape=(num_vertices, num_vertices))
        print('self.observation_space', self.observation_space)
        self.max_steps = 100  # Set the appropriate value for max_steps
        self.myfig = plt.figure()
        self.myax = self.myfig.add_subplot(111)

    def get_outcome(self):
        return self.outcome

    def _get_rewards(self):
        rewards = {
            "player_1": np.sum(self.player1),
            "player_2": np.sum(self.player2)
        }
        return rewards

    def reset(self):
        self.turn = 1
        self.r = 0
        self.outcome = -1
        self.player1 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.player2 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        return self._get_state()

    def get_turn(self):
        return self.turn

    def _get_state(self):
        adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices))
        # Set values based on self.player1
        adjacency_matrix[:self.player1.shape[0], :self.player1.shape[1]] = np.where(self.player1 != 0, 1,
                                                                                    adjacency_matrix[
                                                                                    :self.player1.shape[0],
                                                                                    :self.player1.shape[1]])

        # Set values based on self.player2, considering non-zero values from player1
        adjacency_matrix[:self.player2.shape[0], :self.player2.shape[1]] = np.where(
            (self.player2 != 0) & (adjacency_matrix[:self.player2.shape[0], :self.player2.shape[1]] == 0), 2,
            adjacency_matrix[:self.player2.shape[0], :self.player2.shape[1]])

        return np.stack((self.player1, self.player2, adjacency_matrix), axis=0)

    def draw(self, player, edges):
        max_vertices = (self.num_vertices * (self.num_vertices - 1)) / 2
        if (player == 1):
            if (edges.shape[0] > max_vertices / 2):
                return True
        else:
            if (edges.shape[0] >= max_vertices / 2):
                return True
        return False

    def step(self, action):
        enter_function(inspect.currentframe(), action)

        player = self.turn
        v1, v2 = self._action_to_edge(action)

        if self.player1[v1][v2] != 0 or self.player2[v1][v2] != 0:
            # print("Edge already taken, please enter a different edge")
            self.outcome = -1
            # leave_function(inspect.currentframe(), self._get_state())
            return self._get_state(), -100, True, {'outcome': self.outcome}

        if player == 1:
            self.r = self.r + 1
            self.player1[v1][v2] = self.r
            edges = self.get_edges(self.player1)
            if self.lose(edges, self.player1):
                self._plot_graph(self.player1, self.player2)
                self.myfig.canvas.flush_events()
                print("Player 2 is victorious!")
                self.outcome = 2

                # Save the figure with a timestamp in the file name
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"gameResults/Player1_{timestamp}.png"
                self.myfig.savefig(filename)
                plt.clf()

                #leave_function(inspect.currentframe(), self._get_state())
                return self._get_state(), -100, True, {'outcome': self.outcome}

        else:
            self.player2[v1][v2] = self.r
            edges = self.get_edges(self.player2)
            if self.lose(edges, self.player2):
                self._plot_graph(self.player1, self.player2)
                self.myfig.canvas.flush_events()
                print("Player 1 is victorious!")
                self.outcome = 1

                # Save the figure with a timestamp in the file name
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"gameResults/Player2_{timestamp}.png"
                self.myfig.savefig(filename)
                plt.clf()

                #leave_function(inspect.currentframe(), self._get_state())
                return self._get_state(), 100, True, {'outcome': self.outcome}

        if self.draw(player, edges):
            self._plot_graph(self.player1, self.player2)
            self.myfig.canvas.flush_events()
            print("There are no more unfilled edges. The game has drawn.")
            self.outcome = 0

            # Save the figure with a timestamp in the file name
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"gameResults/Draw_{timestamp}.png"
            self.myfig.savefig(filename)
            plt.clf()

            #leave_function(inspect.currentframe(), self._get_state())
            return self._get_state(), 0, True, {'outcome': self.outcome}

        leave_function(inspect.currentframe(), f'[player, v1, v2, edges, state], {[player, v1, v2, edges, self._get_state()]}')
        self.turn = 2 if self.turn == 1 else 1

        # leave_function(inspect.currentframe(), self._get_state())
        return self._get_state(), self.r - 1, False, {'outcome': self.outcome}

    def get_edges(self, player):
        edges = []
        for v1 in range(player.shape[0]):
            for v2 in range(player.shape[0]):
                if player[v1][v2] != 0:
                    e = np.array([v1, v2])
                    edges.append(e)
        edges = np.array(edges)
        return edges

    def is_clique(self, player, small_clique, vertex):
        for v in small_clique:
            if v == vertex:
                return False
            if player[v, vertex] <= 0 and player[vertex, v] <= 0:
                return False
        return True

    def lose(self, small_clique_set, player):
        clique_set = []
        #enter_function(inspect.currentframe(), f'[player, small_clique_set]{[player, small_clique_set]}')
        for s in small_clique_set:
            for v in range(player.shape[0]):
                if (self.is_clique(player, s, v) == True):
                    new_clique = np.append(s, v)
                    if (len(new_clique) == self.order):

                        #leave_function(inspect.currentframe(), f'new_clique = {new_clique}')
                        return True
                    else:
                        clique_set.append(new_clique)
        clique_set = np.array(clique_set)
        if (np.any(clique_set) == False):
            #leave_function(inspect.currentframe(), f'clique_set = {clique_set}')
            return False
        else:
            #leave_function(inspect.currentframe(), f'clique_set = {clique_set}')
            return self.lose(clique_set, player)

    def render(self, mode="human"):
        #enter_function(inspect.currentframe())

        r = self.player1.max() + self.player2.max() + 1
        self._plot_graph(self.player1, self.player2, r)

        #leave_function(inspect.currentframe())
        # plt.show()

    def _plot_graph(self, player1, player2):
        """
                Plots a graph

                Parameters:
                -----------
                D1 : ndarray
                    distance matrix for player 1f
                D2 : ndarray
                    distance matrix for player 2y
                r : float
                    The parameter for constructing edges in the complex, also the current turn
            """
        # self.myax.clear()  # Clear the previous plot

        G1 = create_simplicial_complex(self.player1, self.r)
        G2 = create_simplicial_complex(self.player2, self.r)

        # plt.figure(figsize = (5, 5))
        pos1 = nx.circular_layout(G1)
        nx.draw_networkx(G1, pos=pos1, with_labels=True, node_size=100, edge_color='red')
        pos2 = nx.circular_layout(G2)
        nx.draw_networkx(G2, pos=pos2, with_labels=True, node_size=100, edge_color='blue')

        # Store the edge labels outside the drawing function
        edge_labels1 = {(u, v): f'{player1[u][v]:.0f}' for u, v in G1.edges}
        edge_labels2 = {(u, v): f'{player2[u][v]:.0f}' for u, v in G2.edges}

        # Update the edge labels dictionaries each time you update the graph
        edge_labels1.update({(u, v): f'{player1[u][v]:.0f}' for u, v in G1.edges})
        edge_labels2.update({(u, v): f'{player2[u][v]:.0f}' for u, v in G2.edges})

        # Draw the edge labels using the updated dictionaries
        nx.draw_networkx_edge_labels(G1, pos=pos1, edge_labels=edge_labels1, font_color='red')
        nx.draw_networkx_edge_labels(G2, pos=pos2, edge_labels=edge_labels2, font_color='blue')

        plt.axis('equal')
        plt.show(block=False)  # Set block=False to allow the program to continue executing

        plt.pause(0.001)  # Add a small pause to refresh the figure window
        # plt.clf()

    def _action_to_edge(self, action):
        #enter_function(inspect.currentframe(), action)

        num_edges = self.num_vertices * (self.num_vertices - 1) // 2
        edge_idx = num_edges - 1 - action

        v1 = 0
        while edge_idx >= self.num_vertices - v1 - 1:
            edge_idx -= self.num_vertices - v1 - 1
            v1 += 1

        v2 = v1 + edge_idx + 1

        #leave_function(inspect.currentframe(), [v1, v2])
        return v1, v2


def train_q_learning_agent(game, num_episodes, learning_starts, discount_factor, epsilon):
    #enter_function(inspect.currentframe())

    env = SimplicialComplexGameWrapper(game)
    env = DummyVecEnv([lambda: env])

    # Initialize the Q-learning agent
    agent = DQN("MlpPolicy", env, learning_starts=learning_starts, gamma=discount_factor, exploration_fraction=epsilon)

    # Train the agent
    agent.learn(total_timesteps=num_episodes * game.max_steps, log_interval=1)

    # Get the final Q-values
    q_values = agent.q_net

    # # Evaluate the agents and get the number of wins
    # agent1_wins, agent2_wins, _ = evaluate_agents(game, q_learning_agent(q_values, epsilon), random_agent,
    #                                               num_eval_episodes)
    #
    # total_episodes = agent.num_timesteps // game.max_steps
    #
    # agent1_win_rate = agent1_wins / total_episodes
    # agent2_win_rate = agent2_wins / total_episodes

    #leave_function(inspect.currentframe())
    return q_values #, agent1_wins, agent2_wins, agent1_win_rate, agent2_win_rate


def q_learning_agent(q_network, epsilon):
    #enter_function(inspect.currentframe(), q_network)

    def agent_fn(state):
        #enter_function(inspect.currentframe())

        # Use the QNetwork to predict the Q-values for the given state
        q_values = q_network.predict(state)[0]
        # print("predict results: ", q_values)

        # Select the action based on epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Randomly select an action
            action = np.random.randint(len(q_values))
        else:
            # Select the action with the highest Q-value
            action = np.argmax(q_values)

        #leave_function(inspect.currentframe(), q_values)
        return action

    #leave_function(inspect.currentframe())
    return agent_fn


def random_agent(state, game):
    # Select a random action from the action space
    #enter_function(inspect.currentframe())
    action = np.random.randint(0, state.shape[1] * (state.shape[2] - 1) // 2)
    edge = game._action_to_edge(action)
    if state[2][edge[0]][edge[1]] != 0:
        return random_agent(state, game)

    #leave_function(inspect.currentframe())
    return action


def evaluate_agents(game, agent1, agent2, num_eval_episodes):
    #enter_function(inspect.currentframe())

    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_badgames = 0
    for gameIndex in range(num_eval_episodes):
        state = game.reset()
        done = False

        while not done:
            player = game.get_turn()
            if player == 1:
                action = agent1(state)
            else:
                action = agent2(state, game)

            state, reward, done, outcome = game.step(action)


        #outcome = game.get_outcome()
        oc = outcome['outcome']
        print("game result = ", oc)
        if oc == 0:
            draws = draws + 1
        elif oc == 1:
            agent1_wins = agent1_wins + 1
        elif oc == 2:
            agent2_wins = agent2_wins + 1
        else:
            total_badgames = total_badgames + 1
            print(f"{total_badgames} game {gameIndex}th is unfinished")

    #leave_function(inspect.currentframe())
    return agent1_wins, agent2_wins, draws


def main():
    # num_vertices = enter_vertices()
    # order = enter_order(num_vertices)
    #enter_function(inspect.currentframe())

    num_vertices = 5
    order = 3

    # Set the random seed
    seed_value = 42
    np.random.seed(seed_value)

    # Create the gameresult folder if it doesn't exist
    if not os.path.exists("gameResults"):
        os.makedirs("gameResults")

    game = SimplicialComplexGame(num_vertices, order)

    # q_values, agent1_wins, agent2_wins, agent1_win_rate, agent2_win_rate = train_q_learning_agent(game,
    #                                                                                               num_episodes=num_episodes,
    #                                                                                               learning_starts=50,
    #                                                                                               discount_factor=0.9,
    #                                                                                               epsilon=epsilon)
    q_values = train_q_learning_agent(game,  num_episodes=num_episodes, learning_starts=1, discount_factor=0.9,
                                                                                                  epsilon=epsilon)
    agent1 = q_learning_agent(q_values, epsilon)
    agent2 = random_agent
    # print("{num_episodes} Training results:")
    # print(f"Agent 1 wins, win_rate: ({agent1_wins}), {agent1_win_rate})")
    # print(f"Agent 2 wins, win_rate: ({agent2_wins}), {agent2_win_rate})")

    agent1_wins, agent2_wins, draws = evaluate_agents(game, agent1, agent2, num_eval_episodes)

    totalgames = agent1_wins + agent2_wins + draws
    print(f"{num_eval_episodes} Evaluation results:")
    if totalgames > 0:
        print(f"Agent 1 wins, win_rate: ({agent1_wins}), {agent1_wins/totalgames})")
        print(f"Agent 2 wins, win_rate: ({agent2_wins}), {agent2_wins/totalgames})")
        print(f"Draws, draw_rate: ({draws}, {draws/totalgames})")

    # Play a game between the trained agent (agent 1) and a random agent (agent 2)
    # state = game.reset()
    # done = False
    #
    # while not done:
    #     player = game.get_turn()
    #     if player == 1:
    #         action = agent1(state)
    #     else:
    #         action = agent2(state)
    #
    #     state, reward, done, _ = game.step(action)
        #print(f"Player {state[2]} placed an edge: {action}")
        #print(f"Current state:\n{state}")

    print("Game over")
    # rewards = game._get_rewards()
    # if rewards["player_1"] == rewards["player_2"]:
    #     print("The game ended in a draw.")
    # elif rewards["player_1"] > rewards["player_2"]:
    #     print("Player 1 is victorious!")
    # else:
    #     print("Player 2 is victorious!")
    # print(f"Final rewards: {rewards}")

    #leave_function(inspect.currentframe())


if __name__ == "__main__":
    main()

