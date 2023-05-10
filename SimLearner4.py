import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import gym
from gym import spaces
#import pyspiel

import numpy as np

import numpy as np
import gym
from gym import spaces

class SimplicialComplexGameWrapper(gym.Wrapper):
    @property
    def unwrapped(self):
        return self.env

    def __init__(self, env):
        super().__init__(env)
        self.metadata = {'render.modes': []}
        self.action_space = spaces.Discrete(self.env.num_valid_moves())
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.env.num_vertices, self.env.num_vertices))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        o = self._get_obs(obs)
        return o, reward, done, info

    def _get_obs(self, obs):
        #adjacency_matrix = self.env._get_state()
        adjacency_matrix = np.zeros((self.env.num_vertices, self.env.num_vertices))
        for i in range(self.env.num_vertices):
            for j in range(i + 1, self.env.num_vertices):
                if self.env._get_state()[0, i, j] != 0:
                    adjacency_matrix[i, j] = 1
                if self.env._get_state()[1, i, j] != 0:
                    adjacency_matrix[i, j] = 2
        return np.reshape(adjacency_matrix, self.observation_space.shape)



from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


def train_q_learning_agent(game, num_episodes, learning_rate, discount_factor, epsilon):
    env = SimplicialComplexGameWrapper(game)
    env = DummyVecEnv([lambda: env])

    # Initialize the Q-learning agent
    agent = DQN("MlpPolicy", env, learning_starts=learning_rate, gamma=discount_factor, exploration_fraction=epsilon)

    # Train the agent
    agent.learn(total_timesteps=num_episodes * game.max_steps, log_interval=1)

    # Get the final Q-values
    q_values = agent.q_net

    print('q_values', q_values)
    return q_values


# def q_learning_agent(q_network):
#     def agent_fn(state):
#         # Convert the state to a flattened array
#         state = np.array(state).flatten()
#
#         # Use the QNetwork to predict the Q-values for the given state
#         q_values = q_network.predict(state)[0]
#
#         # Select the action with the highest Q-value
#         action = np.argmax(q_values)
#
#         return action
#
#     return agent_fn

def q_learning_agent(q_network):
    def agent_fn(state):
        # Convert the state to a flattened array
        #state = np.array(state).flatten()

        # Reshape the state to match the expected shape
        #state = np.reshape(state, (3, 5, 5))
        #print("state: ", state)
        # Use the QNetwork to predict the Q-values for the given state
        q_values = q_network.predict(state)[0]
        #print("predict results: ", q_values)
        # Select the action with the highest Q-value
        action = np.argmax(q_values)
        #print("action = ", action)
        return action

    return agent_fn



def random_agent(state):
    # Select a random action from the action space
    action = np.random.randint(0, state.action_space.n)

    return action


def evaluate_agents(game, agent1, agent2, num_eval_episodes):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    for _ in range(num_eval_episodes):
        state = game.reset()
        done = False

        while not done:
            if state[2].any() == 1:
                action = agent1(state)
            else:
                action = agent2(state)

            state, reward, done, _ = game.step(action)

        rewards = game._get_rewards()
        print("rewards= ", rewards)
        if rewards["player_1"] == rewards["player_2"]:
            draws += 1
        elif rewards["player_1"] > rewards["player_2"]:
            agent1_wins += 1
        else:
            agent2_wins += 1

    return agent1_wins, agent2_wins, draws


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


class SimplicialComplexGame:
    def __init__(self, num_vertices, order):
        self.num_vertices = num_vertices
        self.order = order
        self.turn = 1
        self.r = 1
        self.player1 = np.zeros(shape=(num_vertices, num_vertices))
        self.player2 = np.zeros(shape=(num_vertices, num_vertices))
        self.action_space = spaces.Discrete(self.num_valid_moves())
        print('self.action_space', self.action_space)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, num_vertices, num_vertices))
        print('self.observation_space', self.observation_space)
        self.max_steps = 100  # Set the appropriate value for max_steps

    def reset(self):
        self.turn = 1
        self.r = 1
        self.player1 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.player2 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.action_space = spaces.Discrete(self.num_valid_moves())
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.num_vertices, self.num_vertices))
        return self._get_state()

    def _get_state(self):
        return np.stack((self.player1, self.player2, np.full((self.num_vertices, self.num_vertices), self.turn)),
                        axis=0)

    def _update_action_space(self):
        self.action_space = spaces.Discrete(self.num_valid_moves())

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
        player = self.turn
        print("action= ", action)
        v1, v2 = self._action_to_edge(action)
        print("verticies: ", str(v1), ", ", str(v2))

        if self.player1[v1][v2] != 0 or self.player2[v1][v2] != 0:
            print("Edge already taken, please enter a different edge")
            return self._get_state(), -0.001, False, {}

        if player == 1:
            self.player1[v1][v2] = self.r
            edges = self.get_edges(self.player1)
            if self.lose(edges, self.player1):
                self._plot_graph(self.player1, self.player2)
                print("Player 2 is victorious!")
                return self._get_state(), -1, True, {}

        else:
            self.player2[v1][v2] = self.r
            self.r = self.r + 1
            edges = self.get_edges(self.player2)
            if self.lose(edges, self.player2):
                self._plot_graph(self.player1, self.player2)
                print("Player 1 is victorious!")
                return self._get_state(), 1, True, {}

        if self.draw(player, edges):
            self._plot_graph(self.player1, self.player2)
            print("There are no more unfilled edges. The game has drawn.")
            return self._get_state(), 0, True, {}

        self._update_action_space()
        print('self.action_space', self.action_space)

        self.turn = 2 if self.turn == 1 else 1
        print(self._get_state())
        return self._get_state(), 0, False, {}

    def get_edges(self, player):
        edges = []
        for v1 in range(player.shape[0]):
            for v2 in range(player.shape[0]):
                if player[v1][v2] != 0:
                    e = np.array([v1, v2])
                    edges.append(e)
            edges = np.array(edges)
            return edges

    def get_valid_moves(self):
        unfilled = np.zeros(shape=(self.num_vertices, self.num_vertices))
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if(self.player1[i, j] == 0 and self.player2[i, j] == 0):
                    unfilled[i, j] = 1
        return np.argwhere(unfilled)

    def num_valid_moves(self):
        return self.get_valid_moves().shape[0]

    def is_clique(self, player, small_clique, vertex):
        for v in small_clique:
            if player[v, vertex] <= 0:
                return False
        return True

    def lose(self, small_clique_set, player):
        clique_set = []
        for s in small_clique_set:
            for v in range(player.shape[0]):
                if (self.is_clique(player, s, v) == True and np.any(s == v) == False):
                    new_clique = np.append(s, v)
                    if (len(new_clique) == self.order):
                        return True
                    else:
                        clique_set.append(new_clique)
        clique_set = np.array(clique_set)
        if (np.any(clique_set) == False):
            return False
        else:
            return self.lose(clique_set, player)

    def render(self, mode="human"):
        r = self.player1.max() + self.player2.max() + 1
        self._plot_graph(self.player1, self.player2, r)
        plt.show()

    def _plot_graph(self, player1, player2):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        edges1 = self.get_edges(player1)
        edges2 = self.get_edges(player2)

        for v1, v2 in edges1:
            ax.plot([v1, v2], [v1, v2], 'r-', linewidth=2)

        for v1, v2 in edges2:
            ax.plot([v1, v2], [v1, v2], 'b-', linewidth=2)

        ax.axis('equal')
        ax.axis('off')

        plt.title("Graph")

    def _action_to_edge(self, action):
        v1 = self.get_valid_moves()[action - 1, 0]
        v2 = self.get_valid_moves()[action - 1, 1]
        return v1, v2


def main():

    # num_vertices = enter_vertices()
    # order = enter_order(num_vertices)
    num_vertices=5
    order=3
    game = SimplicialComplexGame(num_vertices, order)

    num_episodes = 1000
    num_eval_episodes = 10

    q_values = train_q_learning_agent(game, num_episodes=num_episodes, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    agent1 = q_learning_agent(q_values)
    agent2 = random_agent


    agent1_wins, agent2_wins, draws = evaluate_agents(game, agent1, agent2, num_eval_episodes)

    print("Evaluation results:")
    print(f"Agent 1 wins: {agent1_wins}")
    print(f"Agent 2 wins: {agent2_wins}")
    print(f"Draws: {draws}")

    # Play a game between the trained agent (agent 1) and a random agent (agent 2)
    state = game.reset()
    done = False

    while not done:
        if state[2].any() == 1:
            action = agent1(state)
        else:
            action = agent2(state)

        state, reward, done, _ = game.step(action)
        print(f"Player {state[2]} placed an edge: {action}")
        print(f"Current state:\n{state}")

    rewards = game._get_rewards()
    print("Game over")
    if rewards[0] == rewards[1]:
        print("The game ended in a draw.")
    elif rewards[0] > rewards[1]:
        print("Player 1 is victorious!")
    else:
        print("Player 2 is victorious!")
    print(f"Final rewards: {rewards}")


if __name__ == "__main__":
    main()

