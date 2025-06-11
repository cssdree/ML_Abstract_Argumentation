from Data.Graphs import GetMINCompletionAttacks
from GNN.Dataset import CreateDGLGraphs
from GNN.Dataset import GetFeatures
from Data.Graphs import WriteApx
from GNN.Training import EGAT
import torch
import sys
import ast
import os

filepath = sys.argv[1]  #chemin du fichier
filepath = os.path.splitext(filepath)[0]  #chemin du fichier sans extension
file = os.path.basename(filepath)  #chemin du fichier sans extension et sans origine
task = sys.argv[2]  #problème de décision
argId = sys.argv[3]  #argument à évaluer

device = "cpu"
cache_root = "cache"
modelpath = "GNN/model/egat_f23_f1.pth"


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
        filepath = os.path.basename(apxpath).replace(".apx", "_MAX.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args + inc_args, def_atts + inc_atts, [], [], filepath)
    else:
        filepath = os.path.basename(apxpath).replace(".apx", "_MIN.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args, GetMINCompletionAttacks(def_atts, inc_args), [], [], filepath)
    return filepath


os.makedirs(cache_root, exist_ok=True)
graph, num_nodes, certain_nodes, is_node_uncertain = CreateDGLGraphs(f"{filepath}.apx")
features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx","MAX"), f"{cache_root}/{file}_MAX.pt")
features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx", "MIN"), f"{cache_root}/{file}_MIN.pt")
node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)

model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
model.eval()
if os.path.exists(modelpath):
    model.load_state_dict(torch.load(modelpath, map_location=device))
with torch.no_grad():
    node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
    predicted = (torch.sigmoid(node_out)>0.5).tolist()
    if task == "PCA":
        if predicted[int(argId)][0] == True:
            print("YES")
        else:
            print("NO")
    elif task == "NCA":
        if predicted[int(argId)][1] == True:
            print("YES")
        else:
            print("NO")
    elif task == "PSA":
        if predicted[int(argId)][2] == True:
            print("YES")
        else:
            print("NO")
    elif task == "NSA":
        if predicted[int(argId)][3] == True:
            print("YES")
        else:
            print("NO")