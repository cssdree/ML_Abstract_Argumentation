from Training import EGAT
import Dataset
import torch
import sys
import ast
import os

file = sys.argv[1]  #chemin du fichier sans extension
task = sys.argv[2]  #problème de décision
argId = sys.argv[3]  #argument à évaluer

model_root = "model"
cache_root = f"{model_root}/cache"
modelpath = f"{model_root}/egat_f23f1.pth"
device = "cpu"


def WriteApx(def_args, def_atts, inc_args, inc_atts, filepath):
    out = open(filepath, "w")
    for arg in def_args:
        out.write(f"arg({arg}).\n")
    for arg in inc_args:
        out.write(f"?arg({arg}).\n")
    for att in def_atts:
        out.write(f"att({att[0]},{att[1]}).\n")
    for att in inc_atts:
        out.write(f"?att({att[0]},{att[1]}).\n")
    out.close()


def GetMINCompletionAttacks(def_atts, inc_args):
    if len(inc_args) == 0:
        return def_atts
    def_atts_MIN = def_atts.copy()
    for inc_arg in inc_args:
        for att in def_atts:
            if inc_arg in att and att in def_atts_MIN:
                def_atts_MIN.remove(att)
    return def_atts_MIN


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
        filepath = apxpath.replace(".apx", "_MAX.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args + inc_args, def_atts + inc_atts, [], [], filepath)
    else:
        filepath = apxpath.replace(".apx", "_MIN.apx")
        filepath = f"{cache_root}/{filepath}"
        WriteApx(def_args, GetMINCompletionAttacks(def_atts, inc_args), [], [], filepath)
    return filepath


os.makedirs(cache_root, exist_ok=True)
graph, num_nodes, certain_nodes, is_node_uncertain = Dataset.CreateDGLGraphs(f"{file}.apx")
features_MAX = Dataset.GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{file}.apx","MAX"), f"{cache_root}/{file}_MAX.pt")
features_MIN = Dataset.GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{file}.apx", "MIN"), f"{cache_root}/{file}_MIN.pt")
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