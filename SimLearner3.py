import numpy as np
import random

class SimplicialComplexGame:
    def __init__(self, num_vertices, order):
        self.num_vertices = num_vertices
        self.order = order
        self.action_space = self.generate_action_space()
        self.simplices = self.generate_simplices()

    def generate_action_space(self):
        actions = []
        for v1 in range(self.num_vertices):
            for v2 in range(v1 + 1, self.num_vertices):
                actions.append((v1, v2))
        return actions

    def generate_simplices(self):
        simplices = []
        for v1 in range(self.num_vertices):
            for v2 in range(v1 + 1, self.num_vertices):
                for v3 in range(v2 + 1, self.num_vertices):
                    simplices.append((v1, v2, v3))
        return simplices

    def is_simplex(self, small_simplex, vertex, distance, r):
        for i in range(len(small_simplex)):
            if small_simplex[i] == vertex or distance[i] > r:
                return False
        return True

    def enter_vertices(self, order, vertex, r, distance, simplices):
        result = []
        if order == 0:
            return [[]]
        for i in range(len(simplices)):
            if simplices[i][0] == vertex and simplices[i][1] < self.num_vertices and simplices[i][2] < self.num_vertices:
                small_simplex = [simplices[i][1], simplices[i][2]]
                if self.is_simplex(small_simplex, vertex, distance, r):
                    rest = self.enter_vertices(order - 1, vertex, r, distance, simplices)
                    for s in rest:
                        s.append(small_simplex)
                    result += rest
        return result

    def enter_order(self, order, r, distance, simplices):
        result = []
        for vertex in range(self.num_vertices):
            rest = self.enter_vertices(order, vertex, r, distance, simplices)
            for s in rest:
                result.append(s)
        return result

    def reset(self):
        state = np.zeros(len(self.action_space))
        return state

    def step(self, action):
        v1, v2 = action
        if (v1, v2, -1) in self.simplices:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        next_state = np.zeros(len(self.action_space))
        return next_state, reward, done, {}

def q_learning_agent(game, q_values, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, len(game.action_space) - 1)
    else:
        state_str = str(state)
        if state_str not in q_values:
            q_values[state_str] = np.zeros(len(game.action_space))
        action = np.argmax(q_values[state_str])
        return action

def random_agent(game, state):
    return random.randint(0, len(game.action_space) - 1)

def train_q_learning_agent(game, num_episodes, learning_rate, discount_factor, epsilon):
    q_values = {}
    for _ in range(num_episodes):
        state = game.reset()
        done = False

        while not done:
            action = q_learning_agent(game, q_values, state, epsilon)
            next_state, reward, done, _ = game.step(game.action_space[action])

            state_str = str(state)
            next_state_str = str(next_state)

            if state_str not in q_values:
                q_values[state_str] = np.zeros(len(game.action_space))

            if next_state_str not in q_values:
                q_values[next_state_str] = np.zeros(len(game.action_space))

            q_values[state_str][action] += learning_rate * (
                    reward + discount_factor * np.max(q_values[next_state_str]) - q_values[state_str][action]
            )

            state = next_state

        return q_values
