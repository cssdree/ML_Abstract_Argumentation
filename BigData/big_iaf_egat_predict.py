from BigData.BigDataset import CreateDGLGraphs
from BigData.BigDataset import GetFeatures
from Data.Graphs import CreateCompletion
from GNN.Training import EGAT
import torch
import sys
import os


def GetAcceptability(model, apxpath, problem, argID):
    model.eval()
    filename = os.path.splitext(os.path.basename(apxpath))[0]
    graph, num_nodes, num_edges, certain_nodes, nodes_id, is_node_uncertain, def_args, inc_args, def_atts, inc_atts, local_device = CreateDGLGraphs(apxpath)
    CreateCompletion(completion, def_args, def_atts, inc_args, inc_atts, f"cache/{filename}.apx")
    if completion == "MIN":
        features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx", local_device)
        node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MIN], dim=1)
    elif completion == "MAX":
        features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx", local_device)
        node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX], dim=1)
    else:
        features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx", local_device)
        features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx", local_device)
        node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
    with torch.no_grad():
        model.to(local_device)
        node_out = model(graph, node_feats, graph.edata["is_uncertain"])
        predicted = (torch.sigmoid(node_out) > 0.5).tolist()
    argID = nodes_id[argID]
    if problem == "PCA":
        if predicted[int(argID)][0] == True:
            return "YES"
        elif predicted[int(argID)][0] == False:
            return "NO"
    elif problem == "NCA":
        if predicted[int(argID)][1] == True:
            return "YES"
        elif predicted[int(argID)][1] == False:
            return "NO"
    elif problem == "PSA":
        if predicted[int(argID)][2] == True:
            return "YES"
        elif predicted[int(argID)][2] == False:
            return "NO"
    elif problem == "NSA":
        if predicted[int(argID)][3] == True:
            return "YES"
        elif predicted[int(argID)][3] == False:
            return "NO"


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    apxpath = sys.argv[1]
    task = sys.argv[2]
    argID = sys.argv[3]
    problem = task.split("-")[0]
    sem = task.split("-")[1]
    completion = task.split("-")[2]
    in_node = 23 if completion == "MINMAX" else 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EGAT(in_node, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(f"GNN/models/egat_f{in_node}_f1_{sem}_{completion}.pth", map_location=device, weights_only=True))
    print(GetAcceptability(model, apxpath, problem, argID))