import numpy as np
import pandas as pd

def enter_game():
    g = input("Enter a game and player: ")
    first = None
    second = None
    for c in range(0, len(g)):
        if (g[c] == ","):
            first = g[:c]
            second = g[c + 1:]

    # check conditions
    if (first == None or second == None):
        print("invalid input: must be 2 natural numbers separated by a comma")
        return enter_game()
    if (first.isnumeric() == False or second.isnumeric() == False):
        print("invalid input: must be numeric")
        return enter_game()
    game = int(first)
    player = int(second)
    if (game < 0 or game >= bottleneck_distances.shape[0] or player < 0 or player >= bottleneck_distances.shape[1]):
        print("invalid input: entered value out of bounds")
        return enter_game()

    return game, player

if __name__ == "__main__":
    bottleneck_distances = pd.read_csv(r'bottleneck_distances/distance_matrix.csv')
    bottleneck_distances = bottleneck_distances.to_numpy()

    game1, player1 = enter_game()
    game2, player2 = enter_game()

    p1 = 3*game1 + player1
    p2 = 3*game2 + player2

    print(bottleneck_distances[p1][p2])