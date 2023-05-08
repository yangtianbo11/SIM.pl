import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pyspiel
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.envs import DummyVecEnv

global num_vertices

def enterVertices():
    """
        user inputs the number of vertices in the game

        Returns
        -------
        v : int
            number of vertices in graph
    """
    vert = input("Number of Vertices: ")
    if(vert.isnumeric() == False):
        print("invalid input: must be a natural number")
        return enterVertices()
    else:
        v = int(vert)
        return v

def enterOrder():
    """
        user inputs the order of the clique that triggers the lose condition

        Returns
        -------
        o : int
            clique order of the lose condition
    """
    ord = input("Order of losing clique: ")
    if(ord.isnumeric() == False):
        print("invalid input: must be a natural number")
        return enterOrder()
    o = int(ord)
    if(o > num_vertices):
        print("invalid input: order must be less than the number of vertices")
        return enterOrder()
    return o

class SimplicialComplexGame(gym.Env):
    def __init__(self, num_vertices, order):
        super(SimplicialComplexGame, self).__init__()
        self.num_vertices = num_vertices
        self.order = order
        self.turn = 1
        self.player1 = np.zeros(shape=(num_vertices, num_vertices))
        self.player2 = np.zeros(shape=(num_vertices, num_vertices))
        self.game = pyspiel.load_game("simplicial_complex")
        self.action_space = spaces.Discrete(num_vertices * (num_vertices - 1) // 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(num_vertices, num_vertices))

    def reset(self):
        self.turn = 1
        self.player1 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.player2 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        return self._get_state()

    def _get_state(self):
        return np.stack((self.player1, self.player2, np.full((self.num_vertices, self.num_vertices), self.turn)), axis=-1)

    def step(self, action):
        player = self.turn
        r = self.player1.max() + self.player2.max() + 1

        v1, v2 = self._action_to_edge(action)

        if self.player1[v1][v2] != 0 or self.player2[v1][v2] != 0:
            print("Edge already taken, please enter a different edge")
            return self._get_state(), 0, False, {}

        if player == 1:
            self.player1[v1][v2] = r
            edges = self.get_edges(self.player1)
            if self.lose(edges, self.player1, r):
                self._plot_graph(self.player1, self.player2, r)
                print("Player 2 is victorious!")
                return self._get_state(), 1, True, {}

        else:
            self.player2[v1][v2] = r
            edges = self.get_edges(self.player2)
            if self.lose(edges, self.player2, r):
                self._plot_graph(self.player1, self.player2, r)
                print("Player 1 is victorious!")
                return self._get_state(), -1, True, {}

        if self.draw(player, edges):
            self._plot_graph(self.player1, self.player2, r)
            print("There are no more unfilled edges. The game has drawn.")
            return self._get_state(), 0, True, {}

        self.turn = 2 if self.turn == 1 else 1
        return self._get_state(), 0, False, {}

    def render(self, mode="human"):
        r = self.player1.max() + self.player2.max() + 1
        self._plot_graph(self.player1, self.player2, r)
        plt.show()

    def close(self):
        pass

    def get_edges(self, player):
        edges = []
        for v1 in range(player.shape[0]):
            for v2 in range(player.shape[0]):
                if player[v1][v2] != 0:
                    e = np.array([v1, v2])
                    edges.append(e)
        edges = np.array(edges)
        return edges

    def is_simplex(self, small_simplex, vertex, distance, r):
        for v in small_simplex:
            if distance[v, vertex] > r or distance[v, vertex] <= 0:
                return False
        return True

    def lose(self, edges, player, r):
        distance = np.zeros((self.num_vertices, self.num_vertices))
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j and player[i][j] != 0:
                    distance[i][j] = distance[j][i] = player[i][j]
                else:
                    distance[i][j] = distance[j][i] = float('inf')

        for v1, v2 in edges:
            for v3 in range(self.num_vertices):
                if v3 != v1 and v3 != v2:
                    small_simplex = [v1, v2, v3]
                    if self.is_simplex(small_simplex, v3, distance, r):
                        return True

        return False

    def draw(self, player, edges):
        if len(edges) == self.num_vertices * (self.num_vertices - 1):
            return True
        return False

    def _plot_graph(self, player1, player2, r):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.num_vertices - 1])
        ax.set_ylim([0, self.num_vertices - 1])

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if player1[i][j] != 0:
                    ax.plot([i, j], [j, i], 'r-')
                if player2[i][j] != 0:
                    ax.plot([i, j], [j, i], 'b-')

        plt.title("Simplicial Complex Game (r = {})".format(r))
        plt.xlabel("Vertices")
        plt.ylabel("Vertices")

        for v in range(self.num_vertices):
            ax.annotate(str(v), xy=(v, v), xytext=(3, 3), textcoords='offset points')

        plt.grid(True)
        plt.draw()

    def _action_to_edge(self, action):
        v1 = 0
        v2 = 0
        edge_idx = action
        count = 0
        for v1 in range(self.num_vertices):
            for v2 in range(v1 + 1, self.num_vertices):
                if self.player1[v1][v2] == 0 and self.player2[v1][v2] == 0:
                    if count == edge_idx:
                        return v1, v2
                    count += 1
        return None

    def _edge_to_action(self, v1, v2):
        count = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.player1[i][j] == 0 and self.player2[i][j] == 0:
                    if i == v1 and j == v2:
                        return count
                    count += 1
        return None

    def train_q_learning_agent(env, num_episodes, learning_rate, discount_factor, epsilon):
        num_actions = env.num_vertices * (env.num_vertices - 1) // 2
        q_values = np.zeros((num_actions, 2))  # Q-values for each action-state pair
        epsilon_decay = epsilon / num_episodes

        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = epsilon_greedy_action(q_values, state, epsilon)
                next_state, reward, done, _ = env.step(action)
                q_values[action][state[2] - 1] += learning_rate * (
                            reward + discount_factor * np.max(q_values[next_state[2] - 1]) -
                            q_values[action][state[2] - 1])
                state = next_state

            epsilon -= epsilon_decay

        return q_values

    def epsilon_greedy_action(q_values, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values[:, state[2] - 1])

    def evaluate_agents(env, agent1, agent2, num_episodes):
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                if state[2] == 1:
                    action = agent1(state)
                else:
                    action = agent2(state)

                state, reward, done, _ = env.step(action)

            if reward == 1:
                agent1_wins += 1
            elif reward == -1:
                agent2_wins += 1
            else:
                draws += 1

        return agent1_wins, agent2_wins, draws

    def random_agent(state):
        legal_actions = state[2]._get_legal_actions()
        return np.random.choice(legal_actions)

    def q_learning_agent(q_values):
        def agent(state):
            legal_actions = state[2]._get_legal_actions()
            action_values = q_values[legal_actions, state[2] - 1]
            return legal_actions[np.argmax(action_values)]

        return agent

    def main():
        num_vertices = enterVertices()
        order = enterOrder()
        game = SimplicialComplexGame(num_vertices, order)

        q_values = train_q_learning_agent(game, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
        agent1 = q_learning_agent(q_values)
        agent2 = random_agent

        num_episodes = 1000
        num_eval_episodes = 100
        agent1_wins, agent2_wins, draws = evaluate_agents(game, agent1, agent2, num_eval_episodes)

        print("Evaluation results:")
        print(f"Agent 1 wins: {agent1_wins}")
        print(f"Agent 2 wins: {agent2_wins}")
        print(f"Draws: {draws}")

        # Play a game between the trained agent (agent 1) and a random agent (agent 2)
        state = game.reset()
        done = False

        while not done:
            if state[2] == 1:
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
