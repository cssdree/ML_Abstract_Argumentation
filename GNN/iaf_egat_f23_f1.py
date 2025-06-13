from Data.Graphs import GetMINCompletionAttacks
from GNN.Dataset import CreateDGLGraphs
from GNN.Dataset import GetFeatures
from Data.Graphs import WriteApx
from GNN.Training import EGAT
import torch
import sys
import ast
import os

modelpath = "GNN/model/egat_f23_f1.pth"
device = "cpu"


def CreateCompletions(apxpath, completion):
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
        filename = os.path.basename(apxpath).replace(".apx", "_MAX")
        filepath = f"cache/{filename}.apx"
        WriteApx(def_args + inc_args, def_atts + inc_atts, [], [], filepath)
    else:
        filename = os.path.basename(apxpath).replace(".apx", "_MIN")
        filepath = f"cache/{filename}.apx"
        WriteApx(def_args, GetMINCompletionAttacks(def_atts, inc_args), [], [], filepath)
    return filepath


def GetAcceptability(model, apxpath, task, argID):
    model.eval()
    filename = os.path.splitext(os.path.basename(apxpath))[0]
    graph, num_nodes, certain_nodes, is_node_uncertain = CreateDGLGraphs(apxpath)
    features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MAX"),f"cache/{filename}_MAX.pt")
    features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MIN"),f"cache/{filename}_MIN.pt")
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
    apxpath = sys.argv[1]  #chemin du fichier AVEC EXTENSION
    task = sys.argv[2]  #problème de décision
    argID = sys.argv[3]  #argument à évaluer
    os.makedirs("cache", exist_ok=True)
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    print(GetAcceptability(model, apxpath, task, argID))