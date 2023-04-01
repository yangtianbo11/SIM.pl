#Authors: Tianbo Yang, Aditya Advani

import numpy as np
import import matplotlib

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


if __name__ == "__main__":
    num_verticies = input("Number of Verticies:")
    dimension = input("Dimension of Hole: ")
