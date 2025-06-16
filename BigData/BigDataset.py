from sklearn.preprocessing import StandardScaler
from dgl.data import DGLDataset
import af_reader_py
import itertools
import torch
import dgl
import os


def CreateDGLGraphs(apxpath, device="cpu"):
    num_nodes = GetNumNodes(apxpath)
    id_counter = itertools.count(start=0)
    nodes_id = {}
    attackers = []
    attacked = []
    certain_nodes = []
    is_node_uncertain = [0]*num_nodes
    is_edge_uncertain = []
    with open(apxpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('arg(') and line.endswith(').'):
                id = next(id_counter)
                nodes_id[str(line[4:-2])] = id
                certain_nodes.append(id)
            elif line.startswith('?arg(') and line.endswith(').'):
                id = next(id_counter)
                nodes_id[str(line[5:-2])] = id
                is_node_uncertain[id] = 1
            elif line.startswith('att(') and line.endswith(').'):
                attackers.append(nodes_id[str(line[4:-2].split(",")[0])])
                attacked.append(nodes_id[str(line[4:-2].split(",")[1])])
                is_edge_uncertain.append(0)
            elif line.startswith('?att(') and line.endswith(').'):
                attackers.append(nodes_id[str(line[5:-2].split(",")[0])])
                attacked.append(nodes_id[str(line[5:-2].split(",")[1])])
                is_edge_uncertain.append(1)
        is_node_uncertain = torch.tensor(is_node_uncertain, dtype=torch.float32).to(device)
        g = dgl.graph((torch.tensor(attackers), torch.tensor(attacked)), num_nodes=num_nodes).to(device)
        g = dgl.add_self_loop(g)
        g.edata["is_uncertain"] = torch.tensor(is_edge_uncertain+[0]*num_nodes, dtype=torch.float32).unsqueeze(1)  #rajout des self loop
    return g, num_nodes, certain_nodes, nodes_id, is_node_uncertain


def GetFeatures(num_nodes, certain_nodes, apxpath, ptpath, device="cpu"):
    if os.path.exists(ptpath):
        raw_features = torch.load(ptpath, map_location="cpu").numpy()
        features = StandardScaler().fit_transform(raw_features)  #normalisation des features
        features = torch.tensor(features, dtype=torch.float32).to(device)
    else:
        raw_features = af_reader_py.compute_features(apxpath, 10000, 0.000001)
        if len(raw_features) != num_nodes:
            raw_features = FullFeatures(num_nodes, certain_nodes, raw_features)
        torch.save(torch.tensor(raw_features, dtype=torch.float32), ptpath)
        features = StandardScaler().fit_transform(raw_features)  #normalisation des features
        features = torch.tensor(features, dtype=torch.float32).to(device)
    return features


def FullFeatures(num_nodes, certain_nodes, raw_features):
    """
    Complète la liste de features de la completion MIN en rajoutant des 0 pour les noeuds manquants
    """
    full_features = [[0]*11 for i in range(num_nodes)]
    for index, node_id in enumerate(certain_nodes):
        for j in range(11):
            full_features[node_id][j] = raw_features[index][j]
    return full_features


def GetNumNodes(apxpath):
    apxfile = os.path.basename(apxpath)
    if apxfile.startswith(('BA','ER','WS','scc','grd','admbuster','stb','sembuster')):
        return int(apxfile.split("_")[1])+1
    else :
        num_nodes = 0
        with open(apxpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (line.startswith('arg(') and line.endswith(').')) or (line.startswith('?arg(') and line.endswith(').')):
                    num_nodes += 1
        return num_nodes