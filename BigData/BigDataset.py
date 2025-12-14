from sklearn.preprocessing import StandardScaler
import af_reader_py
import itertools
import torch
import dgl
import re


def CreateDGLGraphs(apxpath):
    num_nodes = GetNumNodes(apxpath)
    attackers = []
    attacked = []
    certain_nodes = []
    is_node_uncertain = [0]*num_nodes
    is_edge_uncertain = []
    def_args = []
    inc_args = []
    def_atts = []
    inc_atts = []
    nodes_id = {}
    id_counter = itertools.count(start=0)
    pattern = re.compile(r'^(\?)?(arg|att)\((.*?)\).$')
    with open(apxpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if not match:
                continue
            is_uncertain = bool(match.group(1))
            statement = match.group(2)
            content = str(match.group(3))
            if statement == "arg":
                if is_uncertain:
                    inc_args.append(content)
                    id = next(id_counter)
                    nodes_id[content] = id
                    is_node_uncertain[id] = 1
                else:
                    def_args.append(content)
                    id = next(id_counter)
                    nodes_id[content] = id
                    certain_nodes.append(id)
            else:
                src, tgt = content.split(',')
                if is_uncertain:
                    inc_atts.append(tuple([src, tgt]))
                    attackers.append(nodes_id[src])
                    attacked.append(nodes_id[tgt])
                    is_edge_uncertain.append(1)
                else:
                    def_atts.append(tuple([src, tgt]))
                    attackers.append(nodes_id[src])
                    attacked.append(nodes_id[tgt])
                    is_edge_uncertain.append(0)
        num_edges = len(def_atts)+len(inc_atts)
        if num_edges > 1000000:
            local_device = torch.device("cpu")
            print(f"number of edges = {num_edges} -> SWITCHING")
        else:
            local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_node_uncertain = torch.tensor(is_node_uncertain, dtype=torch.float32, device=local_device)
        g = dgl.graph((torch.tensor(attackers, device=local_device), torch.tensor(attacked, device=local_device)), num_nodes=num_nodes, device=local_device)
        g = dgl.add_self_loop(g)
        g.edata["is_uncertain"] = torch.tensor(is_edge_uncertain+[0]*num_nodes, dtype=torch.float32, device=local_device).unsqueeze(1)
    return g, num_nodes, num_edges, certain_nodes, nodes_id, is_node_uncertain, def_args, inc_args, def_atts, inc_atts, local_device


def GetFeatures(num_nodes, certain_nodes, apxpath, device):
    raw_features = af_reader_py.compute_features(apxpath, 10000, 0.000001)
    if len(raw_features) != num_nodes:
        raw_features = FullFeatures(num_nodes, certain_nodes, raw_features)
    features = StandardScaler().fit_transform(raw_features)
    features = torch.tensor(features, dtype=torch.float32, device=device)
    return features


def FullFeatures(num_nodes, certain_nodes, raw_features):
    """
    Complete the feature list of the MIN completion by adding zeros for the missing nodes.
    """
    full_features = [[0]*11 for i in range(num_nodes)]
    for index, node_id in enumerate(certain_nodes):
        for j in range(11):
            full_features[node_id][j] = raw_features[index][j]
    return full_features


def GetNumNodes(apxpath):
    num_nodes = 0
    with open(apxpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (line.startswith('arg(')) or (line.startswith('?arg(')):
                num_nodes += 1
            else:
                return num_nodes