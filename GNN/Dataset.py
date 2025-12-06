from sklearn.preprocessing import StandardScaler
from dgl.data import DGLDataset
import af_reader_py
import numpy
import torch
import dgl
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def CreateDGLGraphs(apxpath, device=device):
    num_nodes = int(os.path.basename(apxpath).split("_")[1])
    attackers = []
    attacked = []
    certain_nodes = []
    is_node_uncertain = [0]*num_nodes
    is_edge_uncertain = []
    def_args = []
    inc_args = []
    def_atts = []
    inc_atts = []
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
                    is_node_uncertain[int(content)] = 1
                else:
                    def_args.append(content)
                    certain_nodes.append(int(content))
            else:
                src, tgt = content.split(',')
                if is_uncertain:
                    inc_atts.append(tuple([src, tgt]))
                    attackers.append(int(src))
                    attacked.append(int(tgt))
                    is_edge_uncertain.append(1)
                else:
                    def_atts.append(tuple([src, tgt]))
                    attackers.append(int(src))
                    attacked.append(int(tgt))
                    is_edge_uncertain.append(0)
        is_node_uncertain = torch.tensor(is_node_uncertain, dtype=torch.float32).to(device)
        g = dgl.graph((torch.tensor(attackers), torch.tensor(attacked)), num_nodes=num_nodes).to(device)
        g = dgl.add_self_loop(g)
        g.edata["is_uncertain"] = torch.tensor(is_edge_uncertain+[0]*num_nodes, dtype=torch.float32).unsqueeze(1).to(device)  #rajout des self loop
    return g, num_nodes, certain_nodes, is_node_uncertain, def_args, inc_args, def_atts, inc_atts


def GetFeatures(num_nodes, certain_nodes, apxpath, ptpath, device=device):
    if os.path.exists(ptpath):
        raw_features = torch.load(ptpath, map_location="cpu", weights_only=True).numpy()
        features = StandardScaler().fit_transform(raw_features)
        features = torch.tensor(features, dtype=torch.float32).to(device)
    else:
        raw_features = af_reader_py.compute_features(apxpath, 10000, 0.000001)
        if len(raw_features) != num_nodes:
            raw_features = FullFeatures(num_nodes, certain_nodes, raw_features)
        torch.save(torch.tensor(raw_features, dtype=torch.float32), ptpath)
        features = StandardScaler().fit_transform(raw_features)
        features = torch.tensor(features, dtype=torch.float32).to(device)
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


def GetLabels(num_nodes, csvpath, device=device):
    label = [[numpy.nan]*4 for i in range(num_nodes)]
    #label matrix of size num_nodes*4 for the four decision problems: PCA, NCA, PSA, NSA
    with open(csvpath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            data = line.strip().split(',')
            for num_problem in range(4):
                label[int(data[0])][num_problem] = int(data[num_problem+1])
    return torch.tensor(label, dtype=torch.float32).to(device)


class Dataset(DGLDataset):
    def __init__(self, IAF_root, sem, completion, device=device):
        self.IAF_root = IAF_root
        self.completions_root = f"{self.IAF_root}/completions"
        self.label_root = f"{self.IAF_root}/labels"
        self.features_root = f"{self.IAF_root}/features"
        self.sem = sem
        self.completion = completion
        self.device = device
        super().__init__(name="Dataset")

    def process(self):
        self.graphs = []
        os.makedirs(self.features_root, exist_ok=True)
        for apxfile in os.listdir(self.IAF_root):
            if apxfile.endswith(".apx"):
                filename = os.path.splitext(apxfile)[0]
                graph, num_nodes, certain_nodes, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(f"{self.IAF_root}/{filename}.apx")
                features_MAX = GetFeatures(num_nodes, certain_nodes, f"{self.completions_root}/{filename}_MAX.apx",f"{self.features_root}/{filename}_MAX.pt")
                features_MIN = GetFeatures(num_nodes, certain_nodes, f"{self.completions_root}/{filename}_MIN.apx",f"{self.features_root}/{filename}_MIN.pt")
                if self.completion == "MIN":
                    features = torch.cat([is_node_uncertain.unsqueeze(1), features_MIN], dim=1)
                elif self.completion == "MAX":
                    features = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX], dim=1)
                else:
                    features = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
                label = GetLabels(num_nodes, f"{self.label_root}/{filename}_{self.sem}.csv")
                mask = ~torch.isnan(label).any(dim=1)  #nodes with all labels marked as valid
                graph.ndata["feat"] = features
                graph.ndata["mask"] = mask
                graph.ndata["label"] = label
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx:int):
        return self.graphs[idx]