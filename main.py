#Authors: Tianbo Yang, Aditya Advani
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#global variables
global num_vertices
global order
global player1
global player2

def VR_simplicial_complex(distance_matrix, r):
    """
    Computes the vertex set, edge set, and triangle set of a 2-dimensional Vietoris-Rips complex from a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray
        The pairwise distance matrix between points.
    r : float
        The parameter for constructing edges in the complex.

    Returns
    -------
    vertex_set : set
        A set containing the indices of the vertices in the complex.
    edge_set : set
        A set containing the indices of the edges in the complex, represented as pairs of vertices.
    triangle_set : set
        A set containing the indices of the triangles in the complex, represented as triples of vertices.
    """
    
    # Create empty sets for vertices, edges, and triangles
    n = distance_matrix.shape[0]
    vertex_set = set(range(n))
    edge_set = set()
    triangle_set = set()
    
    # Add edges and triangles to the complex
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= r:
                edge_set.add(frozenset([i, j]))
                for k in range(j + 1, n):
                    if (distance_matrix[i, k] <= r) and (distance_matrix[j, k] <= r):
                        triangle_set.add(frozenset([i, j, k]))

    return vertex_set, edge_set, triangle_set

def create_simplicial_complex(D,r):
    """
    Input: distance matrix and nonnegative radius
    Output: networkx graph 
    """
    
    G = nx.Graph()
    G.add_nodes_from(list(range(len(D))))
    #edge_list = np.argwhere(D <= r)
    edge_list = getEdges(D)
    G.add_edges_from(edge_list)
    
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

def plotGraph(D1, D2, r):
    """
        Plots a graph

        Parameters:
        -----------
        D1 : ndarray
            distance matrix for player 1
        D2 : ndarray
            distance matrix for player 2
        r : float
            The parameter for constructing edges in the complex, also the current turn
    """
    G1 = create_simplicial_complex(D1, r)
    G2 = create_simplicial_complex(D2, r)

    #plt.figure(figsize = (5, 5))
    nx.draw_networkx(G1, pos = nx.circular_layout(G1), with_labels = True, node_size = 100, edge_color = 'red')
    nx.draw_networkx(G2, pos = nx.circular_layout(G2), with_labels = True, node_size = 100, edge_color = 'blue')

    plt.axis('equal')
    plt.show(block=False)  # Set block=False to allow the program to continue executing

    plt.pause(0.001)  # Add a small pause to refresh the figure window

def enterVertices():
    """
        user inputs the number of verticies in the game

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

def getEdges(distance):
    """
        puts all edges into an array

        Parameters
        ----------
        distance : ndarray
            The pairwise distance matrix between points.

        Returns
        -------
        edges : array
            array containing all pairs of vertices that has a filled edge between them
    """
    edges = []
    for v1 in range(distance.shape[0]):
        for v2 in range(distance.shape[0]):
            if(distance[v1][v2] !=0):
                e = np.array([v1, v2])
                edges.append(e)
    edges = np.array(edges)
    return edges

def isSimplex(smallSimplex, vertex, distance, r):
    """
        detects if there is a n-dimensional simplex

        Parameters
        ----------
        smallSimplex : set
            List of vertices that form a n-1 dimensional simplex
        vertex: int
            vertex to test if it forms a n-dimensional simplex with smallSimplex
        distance : ndarray
            The pairwise distance matrix between points.
        r : float
            The parameter for constructing edges in the complex.

        Returns
        -------
        True if smallSimplex and vertex forms a n-dimensional simplex, false if not
    """
    for v in smallSimplex:
        if(distance[v, vertex] > r or distance[v, vertex] <= 0):
            return False
    return True

def lose(smallSimplexSet, distance, r):
    """
        detects if a player has lost by finding whether there is a d-dimensional simplex

        Parameters
        ----------
        smallSimplexSet : set
            List of n-1 dimensional simplicies
        distance: ndarray
            The pairwise distance matrix between points.
        r : float
            The parameter for constructing edges in the complex.

        Returns
        -------
        True if the simplicial complex contains a d-dimensional simplex, false otherwise
    """
    simplexSet = []
    for s in smallSimplexSet:
        for v in range(distance.shape[0]):
            if(isSimplex(s, v, distance, r) == True and np.any(s == v) == False):
                newSimplex = np.append(s, v)
                if(len(newSimplex) == order):
                    return True
                else:
                    simplexSet.append(newSimplex)
    simplexSet = np.array(simplexSet)
    if(np.any(simplexSet) == False):
        return False
    else:
        return lose(simplexSet, distance, r)

def draw(player, edges):
    """
        detects if the graph is complete and the game is drawn

        Parameters
        ----------
        edges : array
            an array of edges

        Returns
        -------
        True if the graph is complete and false otherwise
    """
    max_vertices = (num_vertices * (num_vertices - 1)) / 2
    if(player == 1):
        if(edges.shape[0] > max_vertices / 2):
            return True
    else:
        if(edges.shape[0] >= max_vertices / 2):
            return True
    return False

import os
import datetime

def takeTurn(player, r):
    """
        Player takes a turn
        Paramenters:
        ------------
        player: int
            the id number of the player currently taking a turn
        r: int
            the current turn of the game
    """


    # Get the current figure and refresh it
    fig = plt.gcf()
    #display graph
    plotGraph(player1, player2, r)
    fig.canvas.flush_events()

    #get string input
    edge = input("Player " + str(player) + " place an edge: ")
    first = None
    second = None
    for c in range(0, len(edge)):
        if(edge[c] == ","):
            first = edge[:c]
            second = edge[c+1:]

    #check conditions
    if(first == None or second == None):
        print("invalid input: edges should be entered in the form of 2 natural numbers separated by a comma")
        return takeTurn(player, r)
    if(first.isnumeric() == False or second.isnumeric() == False):
        print("invalid input: vertices must be numeric")
        return takeTurn(player, r)
    if(int(first) < int(second)):
        v1 = int(first)
        v2 = int(second)
    else:
        v1 = int(second)
        v2 = int(first)
    if(v1 < 0 or v1 >= num_vertices or v2 < 0 or v2 >= num_vertices):
        print("invalid input: entered value out of bounds, not a vertex")
        return takeTurn(player, r)
    if(v1 == v2):
        print("invalid input: vertices must be distinct")
        return takeTurn(player, r)
    if(player1[v1][v2] != 0 or player2[v1][v2] != 0):
        print("edge already taken, please enter a different edge")
        return takeTurn(player, r)

    #add an edge and evaluate lose condition
    if (player == 1):
        player1[v1][v2] = r
        edges = getEdges(player1)
        if(lose(edges, player1, r) == True):
            plotGraph(player1, player2, r)
            print("Player 2 is victorious!")

            # Save the figure with a timestamp in the file name
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"gameResults/game_{timestamp}.png"
            fig.savefig(filename)
            plt.show()
            return 2
        elif (draw(player, edges) == True):
            plotGraph(player1, player2, r)
            print("There are no more unfilled edges. The game has drawn.")
            return 0
        else:
            return takeTurn(2, r)
    else:
        player2[v1][v2] = r
        edges = getEdges(player2)
        if(lose(edges, player2, r) == True):
            plotGraph(player1, player2, r)
            print("Player 1 is victorious!")

            # Save the figure with a timestamp in the file name
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"gameResults/game_{timestamp}.png"
            fig.savefig(filename)
            plt.show()
            return 1
        elif(draw(player, edges) == True):
            plotGraph(player1, player2, r)
            print("There are no more unfilled edges. The game has drawn.")
            return 0
        else:
            r += 1
            return takeTurn(1, r)
        

if __name__ == "__main__":
    num_vertices = enterVertices()
    order = enterOrder()
    turn = 1

    player1 = np.zeros(shape=(num_vertices, num_vertices))
    player2 = np.zeros(shape=(num_vertices, num_vertices))

    # Create the gameresult folder if it doesn't exist
    if not os.path.exists("gameResults"):
        os.makedirs("gameResults")
        
    takeTurn(1, turn)
