import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import pyspiel


def train_q_learning_agent(game, num_episodes, learning_rate, discount_factor, epsilon):
    # Create the Q-learning agent
    agent = pyspiel.QLearner(game, learning_rate, discount_factor, epsilon)

    # Train the agent
    for _ in range(num_episodes):
        agent.evaluate_and_update_until_done()

    # Get the final Q-values
    q_values = agent.q_values()

    return q_values


def q_learning_agent(q_values):
    def agent_fn(state):
        state_str = state.observation_string()
        legal_actions = state.legal_actions()
        q_values_state = q_values[state_str]
        max_q_value = np.max([q_values_state[action] for action in legal_actions])
        best_actions = [action for action in legal_actions if q_values_state[action] == max_q_value]
        action = np.random.choice(best_actions)

        return action

    return agent_fn


def random_agent(state):
    legal_actions = state.legal_actions()
    action = np.random.choice(legal_actions)
    return action


def evaluate_agents(game, agent1, agent2, num_eval_episodes):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    for _ in range(num_eval_episodes):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.current_player() == 0:
                action = agent1(state)
            else:
                action = agent2(state)

            state.apply_action(action)

        rewards = state.rewards()
        if rewards[0] == rewards[1]:
            draws += 1
        elif rewards[0] > rewards[1]:
            agent1_wins += 1
        else:
            agent2_wins += 1

    return agent1_wins, agent2_wins, draws

# from stable_baselines3 import DQN
# from stable_baselines3.common.envs import DummyVecEnv
#
#
# def train_q_learning_agent(game, num_episodes, learning_rate, discount_factor, epsilon):
#     # Create a dummy vectorized environment
#     env = DummyVecEnv([lambda: game])
#
#     # Initialize the Q-learning agent
#     agent = DQN("MlpPolicy", env, learning_rate=learning_rate, gamma=discount_factor, exploration_fraction=epsilon)
#
#     # Train the agent
#     agent.learn(total_timesteps=num_episodes * game.max_steps)
#
#     # Get the final Q-values
#     q_values = agent.q_values
#
#     return q_values
#
#
# def q_learning_agent(q_values):
#     def agent_fn(state):
#         # Convert the state to a flattened array
#         state = np.array(state).flatten()
#
#         # Select the action with the highest Q-value for the given state
#         action = np.argmax(q_values[state])
#
#         return action
#
#     return agent_fn
#
#
# def random_agent(state):
#     # Select a random action from the action space
#     action = np.random.randint(0, state.action_space.n)
#
#     return action
#
#
# def evaluate_agents(game, agent1, agent2, num_eval_episodes):
#     agent1_wins = 0
#     agent2_wins = 0
#     draws = 0
#
#     for _ in range(num_eval_episodes):
#         state = game.reset()
#         done = False
#
#         while not done:
#             if state[2] == 1:
#                 action = agent1(state)
#             else:
#                 action = agent2(state)
#
#             state, reward, done, _ = game.step(action)
#
#         rewards = game._get_rewards()
#         if rewards[0] == rewards[1]:
#             draws += 1
#         elif rewards[0] > rewards[1]:
#             agent1_wins += 1
#         else:
#             agent2_wins += 1
#
#     return agent1_wins, agent2_wins, draws


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

        def render(self, mode="human"):
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


def main():
    num_vertices = enter_vertices()
    order = enter_order(num_vertices)
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

