import os
import networkx as nx
from Graphs import Graphs

methods = {nx.erdos_renyi_graph:"ER", nx.watts_strogatz_graph:"WS", nx.barabasi_albert_graph:"BA"}
nb_graph = 5  #number of graph per settings (without the incomplet part)
nb_nodes = [10,20,30]  #number of nodes
probs_ER = [0.1,0.3,0.5]  #probabilities for erdos renyi
probs_WS = [0.1,0.3,0.5,0.7,0.9]  #probabilities for watts strogatz
nb_edges_BA = [1,2,3]  #number of edges for barabasi albert


def GenerateSeveralIAF(method,nbn,variable,graphs_done):
    """
    Generate several graphs for each probability/edge of each number of nodes for the given method
    """
    for rep in range(nb_graph):
        graph = Graphs(method,nbn,variable)
        graph = CheckSimilarity(graph,graphs_done)
        graphs_done.add(graph.G)
        graph.MakeIncomplet()


def CheckSimilarity(graph,graphs_done):
    """
    Checks that a graph has not already been made
    """
    while graph.G in graphs_done:
        graph = Graphs(graph.method,graph.nbn,graph.var)
    return graph


os.makedirs("IAF_generated", exist_ok=True)
os.makedirs("IAF_generated/completions", exist_ok=True)
graphs_done = set()
for method in methods.keys():
    if method == nx.erdos_renyi_graph:
        for nbn in nb_nodes:
            for variable in probs_ER:
                GenerateSeveralIAF(method,nbn,variable,graphs_done)
    elif method == nx.watts_strogatz_graph:
        for nbn in nb_nodes:
            for variable in probs_WS:
                GenerateSeveralIAF(method,nbn,variable,graphs_done)
    elif method == nx.barabasi_albert_graph:
        for nbn in nb_nodes:
            for variable in nb_edges_BA:
                GenerateSeveralIAF(method,nbn,variable,graphs_done)