from Data.Graphs import CreateCompletions
from GNN.Dataset import CreateDGLGraphs
from GNN.Dataset import GetFeatures
from GNN.Training import EGAT
import torch
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GetAcceptability(model, apxpath, problem, argID):
    model.eval()
    filename = os.path.splitext(os.path.basename(apxpath))[0]
    graph, num_nodes, certain_nodes, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(apxpath)
    len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts, f"cache/{filename}.apx")
    features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx", f"cache/{filename}_MAX.pt")
    features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx", f"cache/{filename}_MIN.pt")
    node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
    with torch.no_grad():
        node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
        predictions = (torch.sigmoid(node_out) > 0.5).tolist()
    if problem == "PCA":
        if predictions[int(argID)][0] == True:
            return "YES"
        elif predictions[int(argID)][0] == False:
            return "NO"
    elif problem == "NCA":
        if predictions[int(argID)][1] == True:
            return "YES"
        elif predictions[int(argID)][1] == False:
            return "NO"
    elif problem == "PSA":
        if predictions[int(argID)][2] == True:
            return "YES"
        elif predictions[int(argID)][2] == False:
            return "NO"
    elif problem == "NSA":
        if predictions[int(argID)][3] == True:
            return "YES"
        elif predictions[int(argID)][3] == False:
            return "NO"


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    apxpath = sys.argv[1]  #chemin du fichier AVEC EXTENSION
    task = sys.argv[2]  #problème de décision - sémantique
    argID = sys.argv[3]  #argument à évaluer
    problem = task.split("-")[0]
    sem = task.split("-")[1]
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(f"GNN/models/egat_f23_f1_{sem}.pth", map_location=device))
    print(GetAcceptability(model, apxpath, problem, argID))