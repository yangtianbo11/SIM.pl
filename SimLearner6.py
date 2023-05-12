import numpy as np
import networkx as nx
from networkx.algorithms import approximation
import random
import inspect
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.vec_env import Wrapper
from stable_baselines3.common.vec_env import VecEnvWrapper


class SimplicialComplexGame(Wrapper):
    def __init__(self, num_nodes, num_edges, num_cliques):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_cliques = num_cliques
        self.G = nx.Graph()
        super().__init__(None)

    def reset(self):
        self.G = nx.Graph()

    def create_graph(self):
        self.G = nx.gnm_random_graph(self.num_nodes, self.num_edges)

    def get_edges(self):
        edges = list(self.G.edges())
        edges = np.array(edges)
        return edges

    def get_cliques(self):
        cliques = list(nx.find_cliques(self.G))
        return cliques

    def get_rewards(self):
        rewards = {}
        cliques = self.get_cliques()
        for clique in cliques:
            for node in clique:
                if node not in rewards:
                    rewards[node] = len(clique)
                else:
                    rewards[node] += len(clique)
        return rewards

    def win(self, clique_set, player):
        for clique in clique_set:
            for node in clique:
                self.G.remove_node(node)

    def lose(self, clique_set, player):
        leave_function(inspect.currentframe())
        return self.lose(clique_set, player)

    def step(self, action):
        return None


def leave_function(frame):
    while frame:
        frame.f_trace = None
        frame = frame.f_back


def train_q_learning_agent(game, num_episodes, learning_starts, discount_factor, epsilon):
    env = DummyVecEnv([lambda: game])
    model = DQN("MlpPolicy", env, learning_starts=learning_starts, verbose=1)
    model.learn(total_timesteps=num_episodes)

    # Evaluate the trained agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    q_values = model.q_net(model.observation_space.sample(), training=False).numpy()

    return q_values, mean_reward


def evaluate_agents(game, agents, num_episodes):
    results = {}

    for agent_name, agent in agents.items():
        env = DummyVecEnv([lambda: game])
        rewards, _ = evaluate_policy(agent, env, n_eval_episodes=num_episodes)
        results[agent_name] = np.mean(rewards)

    return results


def main():
    num_nodes = 50
    num_edges = 200
    num_cliques = 5
    num_episodes = 1000
    learning_starts = 100
    discount_factor = 0.9
    epsilon = 0.1

    game = SimplicialComplexGame(num_nodes, num_edges, num_cliques)
    game.create_graph()

    q_values, mean_reward = train_q_learning_agent(game, num_episodes=num_episodes,
    learning_starts=learning_starts, discount_factor=discount_factor, epsilon=epsilon)

    print("Q-values:", q_values)
    print("Mean reward:", mean_reward)

    agents = {"DQN Agent": DQN("MlpPolicy", DummyVecEnv([lambda: game]))}
    evaluation_results = evaluate_agents(game, agents, num_episodes=10)
    print("Evaluation Results:", evaluation_results)


if __name__ == "__main__":
    main()
