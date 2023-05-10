import numpy as np
import tensorflow as tf
import matplotlib as plt

# Step 1: Define the environment
"""
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        self.current_player = 1  # Player 1 starts the game

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, move):
        self.board[move[0], move[1]] = self.current_player

    def is_winner(self, player):
        rows = np.all(self.board == player, axis=1)
        cols = np.all(self.board == player, axis=0)
        diag1 = np.all(np.diag(self.board) == player)
        diag2 = np.all(np.diag(np.fliplr(self.board)) == player)
        return np.any(rows) or np.any(cols) or diag1 or diag2

    def is_draw(self):
        return np.all(self.board != 0) and not self.is_winner(1) and not self.is_winner(-1)

    def is_terminal(self):
        return self.is_winner(1) or self.is_winner(-1) or self.is_draw()

    def print_board(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' | '.join(symbols[elem] for elem in row))
            print('---------')
"""

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


class simV2:
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
        return self.get_state()

    def get_state(self):
        return np.stack((self.player1, self.player2, np.full((self.num_vertices, self.num_vertices), self.turn)),
                        axis=-1)

    def step(self, action):
        player = self.turn
        r = self.player1.max() + self.player2.max() + 1

        v1, v2 = self.action_to_edge(action)

        if self.player1[v1][v2] != 0 or self.player2[v1][v2] != 0:
            print("Edge already taken, please enter a different edge")
            return self.get_state(), 0, False, {}

        if player == 1:
            self.player1[v1][v2] = r
            edges = self.get_edges(self.player1)
            if self.lose(edges, self.player1, r):
                self.plot_graph(self.player1, self.player2, r)
                print("Player 2 is victorious!")
                return self.get_state(), 1, True, {}

        else:
            self.player2[v1][v2] = r
            edges = self.get_edges(self.player2)
            if self.lose(edges, self.player2, r):
                self.plot_graph(self.player1, self.player2, r)
                print("Player 1 is victorious!")
                return self.get_state(), -1, True, {}

        if self.draw(player, edges):
            self.plot_graph(self.player1, self.player2, r)
            print("There are no more unfilled edges. The game has drawn.")
            return self.get_state(), 0, True, {}

        self.turn = 2 if self.turn == 1 else 1
        return self.get_state(), 0, False, {}

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
            self.plot_graph(self.player1, self.player2, r)
            plt.show()

        def plot_graph(self, player1, player2, r):
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


# Step 2: Set up the RL agent

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.state_size,), activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def update_model(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(next_state)[0]
            target += self.discount_factor * np.max(next_q_values)
        target_q_values = self.model.predict(state)
        target_q_values[0][action] = target
        self.model.fit(state, target_q_values, epochs=1, verbose=0)

# Step 3: Implement the RL algorithm

def q_learning(agent, env, episodes, epsilon, epsilon_decay, epsilon_min):
    for episode in range(episodes):
        state = env.board.flatten()
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            valid_moves = env.get_valid_moves()
            move = valid_moves[action]

            env.make_move(move)

            next_state = env.board.flatten()

            if env.is_winner(1):
                reward = 1
                done = True
            elif env.is_winner(-1):
                reward = -1
                done = True
            elif env.is_draw():
                reward = 0
                done = True
            else:
                reward = 0

            agent.update_model(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Step 4: Training loop
num_vertices = enter_vertices()
order = enter_order(num_vertices)
game = simV2(num_vertices, order)

agent = QAgent(state_size=9, action_size=game.get_valid_moves().shape[0], learning_rate=learning_rate, discount_factor=discount_factor)

q_learning(agent, game, episodes, epsilon, epsilon_decay, epsilon_min)

# Step 5: Evaluation
def play_game(agent):
    game = simV2(num_vertices, order)
    done = False
    while not done:
        state = game.board.flatten()
        action = agent.get_action(state, epsilon=0)  # No exploration
        valid_moves = game.get_valid_moves()
        move = valid_moves[action]
        game.make_move(move)
        game.print_board()
        print()

        if game.is_winner(1):
            print("Agent wins!")
            done = True
        elif game.is_winner(-1):
            print("Opponent wins!")
            done = True
        elif game.is_draw():
            print("It's a draw!")
            done = True

play_game(agent)
