import main
from main import enterVertices, enterOrder, getEdges, lose, draw
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import random
import sys
import threading

"""
global num_vertices
global order
global player1
global player2
"""

def simulate(player, r):
    """
        Player takes a turn
        Paramenters:
        ------------
        player: int
            the id number of the player currently taking a turn
        r: int
            the current turn of the game
    """

    #get edge
    v1 = random.randrange(0, main.num_vertices)
    v2 = random.randrange(0, main.num_vertices)

    #check conditions
    if(v1 > v2):
        temp = v1
        v1 = v2
        v2 = temp
    if(v1 == v2):
        return simulate(player, r)
    if(main.player1[v1][v2] != 0 or main.player2[v1][v2] != 0):
        return simulate(player, r)

    print("(" + str(v1) + "," + str(v2) + ")")

    #add an edge and evaluate lose condition
    if (player == 1):
        main.player1[v1][v2] = r
        edges = getEdges(main.player1)
        if(lose(edges, main.player1, r) == True):
            print("Player 2 is victorious!")
            return 2
        elif (draw(player, edges) == True):
            print("There are no more unfilled edges. The game has drawn.")
            return 0
        else:
            return simulate(2, r)
    else:
        main.player2[v1][v2] = r
        edges = getEdges(main.player2)
        if(lose(edges, main.player2, r) == True):
            print("Player 1 is victorious!")
            return 1
        elif(draw(player, edges) == True):
            print("There are no more unfilled edges. The game has drawn.")
            return 0
        else:
            r += 1
            return simulate(1, r)

if __name__ == "__main__":
    sys.setrecursionlimit(10 ** 7)  # max depth of recursion
    threading.stack_size(2 ** 27)  # new thread will get stack of such size

    main.num_vertices = enterVertices()
    main.order = enterOrder()

    for i in range(50):
        turn = 1

        main.player1 = np.zeros(shape=(main.num_vertices, main.num_vertices))
        main.player2 = np.zeros(shape=(main.num_vertices, main.num_vertices))

        #simulation
        result = simulate(1, turn)
        combined = main.player1 + main.player2

        # plot persistent diagrams
        if result == 1:
            dgms1 = ripser(main.player1, maxdim=2)['dgms']
            fig1 = plt.figure(figsize=(6,6))
            plot_diagrams(dgms1, show=False)
            plt.savefig("persistent_diagrams/player1_wins/player1_persistent_diagram" + str(i) + ".png")
            plt.close(fig1)

            dgms2 = ripser(main.player2, maxdim=2)['dgms']
            fig2 = plt.figure(figsize=(6,6))
            plot_diagrams(dgms2, show=False)
            plt.savefig("persistent_diagrams/player1_wins/player2_persistent_diagram" + str(i) + ".png")
            plt.close(fig2)

            dgms3 = ripser(combined, maxdim=2)['dgms']
            fig3 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms3, show=False)
            plt.savefig("persistent_diagrams/player1_wins/combined_persistent_diagram" + str(i) + ".png")
            plt.close(fig3)
        elif result == 2:
            dgms1 = ripser(main.player1, maxdim=2)['dgms']
            fig1 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms1, show=False)
            plt.savefig("persistent_diagrams/player2_wins/player1_persistent_diagram" + str(i) + ".png")
            plt.close(fig1)

            dgms2 = ripser(main.player2, maxdim=2)['dgms']
            fig2 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms2, show=False)
            plt.savefig("persistent_diagrams/player2_wins/player2_persistent_diagram" + str(i) + ".png")
            plt.close(fig2)

            dgms3 = ripser(combined, maxdim=2)['dgms']
            fig3 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms3, show=False)
            plt.savefig("persistent_diagrams/player2_wins/combined_persistent_diagram" + str(i) + ".png")
            plt.close(fig3)
        elif result == 0:
            dgms1 = ripser(main.player1, maxdim=2)['dgms']
            fig1 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms1, show=False)
            plt.savefig("persistent_diagrams/draws/player1_persistent_diagram" + str(i) + ".png")
            plt.close(fig1)

            dgms2 = ripser(main.player2, maxdim=2)['dgms']
            fig2 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms2, show=False)
            plt.savefig("persistent_diagrams/draws/player2_persistent_diagram" + str(i) + ".png")
            plt.close(fig2)

            dgms3 = ripser(combined, maxdim=2)['dgms']
            fig3 = plt.figure(figsize=(6, 6))
            plot_diagrams(dgms3, show=False)
            plt.savefig("persistent_diagrams/draws/combined_persistent_diagram" + str(i) + ".png")
            plt.close(fig3)
