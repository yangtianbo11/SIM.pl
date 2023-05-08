import numpy as np
import matplotlib.pyplot as plt
import gym
import pyspiel

class simV2(gym.Env):
    def __init__(self, num_vertices, order):
        super(simV2, self).__init__()
        self.num_vertices = num_vertices
        self.order = order
        self.turn = 1
        self.player1 = np.zeros(shape=(num_vertices, num_vertices))
        self.player2 = np.zeros(shape=(num_vertices, num_vertices))
        self.game = pyspiel.load_game("simV2")

    def reset(self):
        self.turn = 1
        self.player1 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        self.player2 = np.zeros(shape=(self.num_vertices, self.num_vertices))
        return self._get_state()

    def _get_state(self):
        return self.player1, self.player2, self.turn

    def step(self, action):
        player = self.turn
        r = self.player1.max() + self.player2.max() + 1

        v1, v2 = action

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

    def _get_state(self):
        return self.player1, self.player2, self.turn

    def _is_terminal(self):
        edges1 = self._get_edges(self.player1)
        edges2 = self._get_edges(self.player2)

        if self.lose(edges1, self.player1, self.turn):
            return True
        if self.lose(edges2, self.player2, self.turn):
            return True
        if self.draw(self.player1, edges1) or self.draw(self.player2, edges2):
            return True

        return False

    def _get_legal_actions(self):
        legal_actions = []
        for v1 in range(self.num_vertices):
            for v2 in range(v1 + 1, self.num_vertices):
                if self.player1[v1][v2] == 0 and self.player2[v1][v2] == 0:
                    legal_actions.append((v1, v2))
        return legal_actions

    def _apply_action(self, action):
        v1, v2 = action
        if self.turn == 1:
            self.player1[v1][v2] = self.turn
        else:
            self.player2[v1][v2] = self.turn

        self.turn = 3 - self.turn

    def _get_rewards(self):
        if self.lose(self._get_edges(self.player1), self.player1, self.turn):
            return [-1, 1]
        if self.lose(self._get_edges(self.player2), self.player2, self.turn):
            return [1, -1]
        if self.draw(self.player1, self._get_edges(self.player1)) or self.draw(self.player2,
                                                                               self._get_edges(self.player2)):
            return [0, 0]
        return None

        def _clone(self):
            return simV2(self.num_vertices, self.order, self.player1.copy(), self.player2.copy(),
                                         self.turn)

    # end of simV2 class

    def main():
        num_vertices = enterVertices()
        order = enterOrder()
        game = simV2(num_vertices, order)

        state = game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            state.apply_action(action)
            print(f"Player {player} placed an edge: {action}")
            print(f"Current state:\n{state}")

        rewards = state.rewards()
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
