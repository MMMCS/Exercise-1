import numpy as np
import networkx as nx


def huckel(problem_type, nodes_or_3d_problem, alpha, beta):

    """ This code takes the following code inputs:
        problem_type: specify "2d_linear", "2d_cyclic", "3d" for the geometry of respective problems
        nodes_or_3d_problem: specify "tetra", "cubic", "icosa", "dodeca", "octa", "bucky" for respective 3D problems (to save node counting), specify number of nodes for 2D problems
        alpha: specify huckel theory alpha parameter
        beta: specify huckel theory beta parameter
        The code then outputs the huckel energy levels with associated degeneracies in a text file.
        This code requires numpy and networkx to be installed."""

    # Generate Huckel matrix for 3D problem
    if problem_type == "3d":
        if nodes_or_3d_problem == "tetra":
            graph = nx.tetrahedral_graph()  # Generates graph for 3D shape specified
            nodes = 4  # Node count for 3D problem called
        if nodes_or_3d_problem == "cubic":
            graph = nx.cubical_graph()
            nodes = 8
        if nodes_or_3d_problem == "icosa":
            graph = nx.icosahedral_graph()
            nodes = 12
        if nodes_or_3d_problem == "octa":
            graph = nx.octahedral_graph()
            nodes = 6
        if nodes_or_3d_problem == "dodeca":
            graph = nx.dodecahedral_graph()
            nodes = 20
        if nodes_or_3d_problem == "bucky":  # Hard code graph for C60 bucky molecule
            nodes = 60
            A, B, C1, C2, D1, D2, E1, E2, F1, F2, G, H = [], [], [], [], [], [], [], [], [], [], [], []  # Initialise categories of nodes
            for node_label in range(0, 5):  # Generate nodes
                A.append(node_label)
            for node_label in range(5, 10):
                B.append(node_label)
            for node_label in range(10, 15):
                C1.append(node_label)
            for node_label in range(15, 20):
                C2.append(node_label)
            for node_label in range(20, 25):
                D1.append(node_label)
            for node_label in range(25, 30):
                D2.append(node_label)
            for node_label in range(30, 35):
                E1.append(node_label)
            for node_label in range(35, 40):
                E2.append(node_label)
            for node_label in range(40, 45):
                F1.append(node_label)
            for node_label in range(45, 50):
                F2.append(node_label)
            for node_label in range(50, 55):
                G.append(node_label)
            for node_label in range(55, 60):
                H.append(node_label)
            A_reorder = [1, 2, 3, 4, 0]
            A_shifted = [A[node_label] for node_label in A_reorder]
            H_shifted = [H[node_label] for node_label in A_reorder]

            edges_list = []  # Generate edges between generated nodes
            for node_label in range(0, 5):
                edges_list.append((A[node_label], A_shifted[node_label]))
                edges_list.append((A[node_label], B[node_label]))
                edges_list.append((B[node_label], C1[node_label]))
                edges_list.append((B[node_label], C2[node_label]))
                edges_list.append((C1[node_label], C2[node_label]))
                edges_list.append((C1[node_label], D1[node_label]))
                edges_list.append((C2[node_label], D2[node_label]))
                edges_list.append((D1[node_label], D2[node_label]))
                edges_list.append((D1[node_label], E1[node_label]))
                edges_list.append((D2[node_label], E2[node_label]))
                edges_list.append((E1[node_label], E2[node_label]))
                edges_list.append((E1[node_label], F1[node_label]))
                edges_list.append((E2[node_label], F2[node_label]))
                edges_list.append((F1[node_label], F2[node_label]))
                edges_list.append((F1[node_label], G[node_label]))
                edges_list.append((F2[node_label], G[node_label]))
                edges_list.append((G[node_label], H[node_label]))
                edges_list.append((H[node_label], H_shifted[node_label]))

            graph = nx.Graph()  # Initialise bucky graph
            graph.add_edges_from(edges_list)  # Generate bucky graph

        node_list = list(range(0, nodes))  # List node numbers to remove from adjacency matrix below with rc_order
        adjacency_matrix = nx.attr_matrix(graph, rc_order = node_list)  # Generate adjacency matrix based on connected nodes
        huckel_matrix = alpha * np.identity(nodes) + beta * adjacency_matrix   # Generate huckel matrix using alpha and beta parameters

    # Generate Huckel matrix for 2D problems
    if problem_type == "2d_linear":  # 2D linear problem
        huckel_matrix = np.zeros((nodes_or_3d_problem, nodes_or_3d_problem))  # Initialise matrix of correct dimensionality
        for node_label in range(nodes_or_3d_problem):  # Iterate over nodes to generate huckel matrix
            huckel_matrix[node_label, node_label] = alpha  # Diagonal elements are alpha
            if (node_label - 1) >= 0:  # Adjacent nodes assigned beta
                huckel_matrix[node_label, node_label - 1] = beta
            if (node_label + 1) <= (nodes_or_3d_problem - 1):
                huckel_matrix[node_label, node_label + 1] = beta

    if problem_type == "2d_cyclic":  # 2D cyclic problem
        huckel_matrix = np.zeros((nodes_or_3d_problem, nodes_or_3d_problem))
        for node_label in range(nodes_or_3d_problem):  # Iterate over nodes to generate huckel matrix
            huckel_matrix[0, nodes_or_3d_problem - 1] = beta  # Adjacent nodes assigned beta
            huckel_matrix[nodes_or_3d_problem - 1, 0] = beta
            huckel_matrix[node_label, node_label] = alpha  # Diagonal elements are alpha
            if (node_label - 1) >= 0:  # Adjacent nodes assigned beta
                huckel_matrix[node_label, node_label - 1] = beta
            if (node_label + 1) <= (nodes_or_3d_problem - 1):
                huckel_matrix[node_label, node_label + 1] = beta

    # Solve for huckel energies
    eigenvalues, eigenvectors = np.linalg.eig(huckel_matrix)
    energies = (sorted(eigenvalues.real))  # Take real part as numerical methods can generate mistaken complex solution

    # Counts degeneracies of energies
    temporary_list = []
    degeneracies = []
    for given_energy in range(len(energies)):  # For each energy, loop searches for degenerate energies that are only slightly different due only to numerical methods
        for i in range(len(energies)):
            temporary_list.append(np.isclose(energies[given_energy],energies[i],rtol=1e-06, atol=1e-06, equal_nan=True))  # Creates a list with "True" for degenerate energy levels, else "False"
        degeneracies.append(temporary_list.count(True))  # counts "True"s and adds count to list of the degeneneracies for given energy
        temporary_list = []  # Clears temporary list for next iteration

    # Remove repeated energies (with degeneracy greater than 1)
    for given_energy in range(len(energies)-1):  # Loop removes repeats in the list of energies and degeneracies, uses fact that energies are in size order
        if np.isclose(energies[given_energy], energies[given_energy+1], rtol=1e-06, atol=1e-06, equal_nan=True):  # Compare consecutive energies in list
            energies[given_energy] = "Remove"  # If energy is repeated replace given energy with "Remove"
            degeneracies[given_energy] = "Remove" # If energy is repeated replace given energy's degeneracy with "Remove"
    energies[:] = (value for value in energies if value != "Remove")  # Removes duplicates represented by "Remove"
    degeneracies[:] = (value for value in degeneracies if value != "Remove")
    rounded_energies = [round(elem, 4) for elem in energies]    # round float to manageable size for presentation

    # Output energies and degeneracies in text file
    print(" Problem Type:", problem_type, "\n", "Number of nodes or type of 3D problem:", nodes_or_3d_problem, "\n", "Alpha:", alpha, "\n", "Beta:", beta,
          "\n", "Energy:", rounded_energies, "\n", "Degeneracy", degeneracies,  file=open("Huckel_Output.txt", "w"))

huckel("3d", "tetra", 0, -1)