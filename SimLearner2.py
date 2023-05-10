import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from open_spiel.python.algorithms import random_agent
#from open_spiel.python.algorithms import tabular_qlearner as q_learning
#from open_spiel.python import policy

class SimplicialComplexGame:
    def __init__(self, num_vertices, order):
        self.num_vertices = num_vertices
        self.order = order
        self.turn = 1
        self.player1 = np.zeros(shape=(num_vertices, num_vertices))
        self.player2 = np.zeros(shape=(num_vertices, num_vertices))
        self.action_space = spaces.Discrete(num_vertices * (num_vertices - 1) // 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(num_vertices, num_vertices))

    def reset(self):
        self.turn = 1
        self.player1 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.player2 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        return self._get_state()

    def _get_state(self):
        return np.stack((self.player1, self.player2, np.full((self.num_vertices, self.num_vertices), self.turn)),
                        axis=-1)

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

    def get_winner(self):
        if self.player1_wins:
            return 1
        elif self.player2_wins:
            return 2
        else:
            return 0

    def render(self):
        r = self.player1.max() + self.player2.max() + 1
        self._plot_graph(self.player1, self.player2, r)
        plt.show()

    def _plot_graph(self, player1, player2, r):
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
        plt.show()

    def lose(self, edges, player, r):
        """Check if a player has lost the game.
        Args:
            edges: A numpy array of shape (num_edges, 2) representing the edges of the graph.
            player: A numpy array representing the player's graph.
            r: An integer representing the maximum distance to check.
        Returns:
            True if the player has lost, False otherwise.
        """
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


    def render(self, mode="human"):
        """Render the current state of the game.
        Args:
            mode: The rendering mode. Currently supports "human" for displaying in a window.
        """
        r = self.player1.max() + self.player2.max() + 1
        self._plot_graph(self.player1, self.player2, r)
        plt.show()


    def _plot_graph(self, player1, player2, r):
        """Plot the graph based on the players' graphs.
        Args:
            player1: A numpy array representing player 1's graph.
            player2: A numpy array representing player 2's graph.
            r: An integer representing the maximum distance to check.
        """
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
        plt.show()

def enter_vertices():
    while True:
        try:
            num_vertices = int(input("Enter the number of vertices: "))
            if num_vertices <= 0:
                print("Number of vertices must be a positive integer.")
            else:
                return num_vertices
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of vertices.")

def enter_order():
    while True:
        try:
            order = int(input("Enter the order: "))
            if order <= 0:
                print("Order must be a positive integer.")
            else:
                return order
        except ValueError:
            print("Invalid input. Please enter a valid integer for the order.")

import numpy as np
import random

def train_q_learning_agent(game, num_episodes, learning_rate, discount_factor, epsilon):
    # Initialize the Q-values table as a dictionary
    q_values = {}

    for _ in range(num_episodes):
        state = game.reset()
        done = False

        while not done:
            # Convert the state to a string to use as a dictionary key
            state_str = str(state)

            if state_str not in q_values:
                # Add the state to the Q-values table if it's encountered for the first time
                q_values[state_str] = {}

            if random.random() < epsilon:
                # Select a random action with probability epsilon
                action = random.randint(0, game.action_space.n - 1)
            else:
                # Select the action with the highest Q-value for the given state
                if q_values[state_str]:
                    action = max(q_values[state_str], key=q_values[state_str].get)
                else:
                    action = random.randint(0, game.action_space.n - 1)

            next_state, reward, done, _ = game.step(action)

            if str(next_state) not in q_values:
                # Add the next state to the Q-values table if it's encountered for the first time
                q_values[str(next_state)] = {}

            if action not in q_values[state_str]:
                # Initialize the Q-value for the (state, action) pair
                q_values[state_str][action] = 0.0

            if not done:
                # Update the Q-value for the (state, action) pair
                q_values[state_str][action] += learning_rate * (
                        reward + discount_factor * max(q_values[str(next_state)].values()) - q_values[state_str][action])
            else:
                # Terminal state has no next state Q-value
                q_values[state_str][action] += learning_rate * (reward - q_values[state_str][action])

            state = next_state

    return q_values


def q_learning_agent(q_values):
    def agent_fn(state):
        state_str = str(state)
        if state_str in q_values:
            # Select the action with the highest Q-value for the given state
            action = max(q_values[state_str], key=q_values[state_str].get)
        else:
            # Select a random action if the state is not in the Q-values table
            action = random.randint(0, game.action_space.n - 1)

        return action

    return agent_fn

# Rest of the code...


def random_agent(game):
    return random.Agent(game)

def evaluate_agents(game, q_agent, random_agent, num_episodes):
    q_agent_wins = 0
    random_agent_wins = 0

    for _ in range(num_episodes):
        current_state = game.new_initial_state()
        while not current_state.is_terminal():
            if current_state.current_player() == 0:
                action = q_agent.step(current_state)
            else:
                action = random_agent.step(current_state)
            current_state.apply_action(action)

        winner = current_state.get_winner()
        if winner == 0:
            random_agent_wins += 1
        elif winner == 1:
            q_agent_wins += 1

    return q_agent_wins, random_agent_wins

def main():
    num_vertices = enter_vertices()
    order = enter_order()
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
            # Agent 1's turn
            action = agent1.act(state)
        else:
            # Agent 2's turn
            action = agent2.act(state)

        state, _, done, _ = game.step(action)
        game.render()

        # Determine the winner of the game
    winner = game.get_winner()
    if winner == 1:
        print("Agent 1 wins!")
    elif winner == 2:
        print("Agent 2 wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
