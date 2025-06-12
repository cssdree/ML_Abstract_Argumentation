from Data.Graphs import GetMINCompletionAttacks
from GNN.Dataset import CreateDGLGraphs
from GNN.Dataset import GetFeatures
from Data.Graphs import WriteApx
from GNN.Training import EGAT
import torch
import sys
import ast
import os

device = "cpu"
cache_root = "cache"
modelpath = "GNN/model/egat_f23_f1.pth"


def CreateCompletions(apxpath, cache_root, completion):
    inc_args = []
    inc_atts = []
    def_args = []
    def_atts = []
    with open(apxpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('arg(') and line.endswith(').'):
                def_args.append(int(line[4:-2]))
            elif line.startswith('?arg(') and line.endswith(').'):
                inc_args.append(int(line[5:-2]))
            elif line.startswith('att(') and line.endswith(').'):
                def_atts.append(ast.literal_eval(line[3:-1]))
            elif line.startswith('?att(') and line.endswith(').'):
                inc_atts.append(ast.literal_eval(line[4:-1]))
    if completion == "MAX":
        filepath = os.path.basename(apxpath).replace(".apx", "_MAX.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args + inc_args, def_atts + inc_atts, [], [], filepath)
    else:
        filepath = os.path.basename(apxpath).replace(".apx", "_MIN.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args, GetMINCompletionAttacks(def_atts, inc_args), [], [], filepath)
    return filepath


def GetAcceptability(model, cache_root, filepath,task,argID):
    model.eval()
    filename = os.path.basename(filepath)  #chemin du fichier sans extension et sans origine
    graph, num_nodes, certain_nodes, is_node_uncertain = CreateDGLGraphs(f"{filepath}.apx")
    features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx", cache_root, "MAX"),f"{cache_root}/{filename}_MAX.pt")
    features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx", cache_root, "MIN"),f"{cache_root}/{filename}_MIN.pt")
    node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
    with torch.no_grad():
        node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
        predicted = (torch.sigmoid(node_out) > 0.5).tolist()
        if task == "PCA":
            if predicted[int(argID)][0] == True:
                return "YES"
            elif predicted[int(argID)][0] == False:
                return "NO"
        elif task == "NCA":
            if predicted[int(argID)][1] == True:
                return "YES"
            elif predicted[int(argID)][1] == False:
                return "NO"
        elif task == "PSA":
            if predicted[int(argID)][2] == True:
                return "YES"
            elif predicted[int(argID)][2] == False:
                return "NO"
        elif task == "NSA":
            if predicted[int(argID)][3] == True:
                return "YES"
            elif predicted[int(argID)][3] == False:
                return "NO"


if __name__ == "__main__":
    filepath = sys.argv[1]  #chemin du fichier sans extension
    task = sys.argv[2]  #problème de décision
    argID = sys.argv[3]  #argument à évaluer
    os.makedirs(cache_root, exist_ok=True)
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    print(GetAcceptability(model, filepath, task, argID))