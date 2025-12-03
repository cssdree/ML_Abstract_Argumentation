from Data.Graphs import Graphs
from CONFIG import IAF_root
import networkx as nx
import numpy as np
import random
import os

methods = [nx.erdos_renyi_graph, nx.watts_strogatz_graph, nx.barabasi_albert_graph]
nb_nodes = [10,20,30]  #number of nodes
probs_ER = [0.1,0.3,0.5]  #probabilities for erdos renyi
probs_WS = [0.1,0.3,0.5,0.7,0.9]  #probabilities for watts strogatz
nb_edges_BA = [1,2,3]  #number of edges for barabasi albert

seed_train = 123
seed_test = 456

if IAF_root == "Data/IAF_TrainSet":
    nb_graphs = 5
    np.random.seed(seed_train)
    random.seed(seed_train)
if IAF_root == "Data/IAF_TestSet":
    nb_graphs = 2
    np.random.seed(seed_test)
    random.seed(seed_test)


def GenerateSeveralIAF(method,nbn,variable):
    """
    Generate several graphs for each probability/edge of each number of nodes for the given method
    """
    for rep in range(nb_graphs):
        graph = Graphs(method,nbn,variable,IAF_root)
        graph = CheckSimilarity(graph)
        graphs_done.add(graph.G)
        graph.MakeIncomplet()


def CheckSimilarity(graph):
    """
    Check that a graph has not already been made
    """
    while graph.G in graphs_done:
        graph = Graphs(graph.method,graph.nbn,graph.var,IAF_root)
    return graph


if __name__ == "__main__":
    os.makedirs(IAF_root, exist_ok=True)
    os.makedirs(f"{IAF_root}/completions", exist_ok=True)
    graphs_done = set()
    for method in methods:
        if method == nx.erdos_renyi_graph:
            for nbn in nb_nodes:
                for variable in probs_ER:
                    GenerateSeveralIAF(method,nbn,variable)
        elif method == nx.watts_strogatz_graph:
            for nbn in nb_nodes:
                for variable in probs_WS:
                    GenerateSeveralIAF(method,nbn,variable)
        elif method == nx.barabasi_albert_graph:
            for nbn in nb_nodes:
                for variable in nb_edges_BA:
                    GenerateSeveralIAF(method,nbn,variable)