from BigData.BigDataset import CreateDGLGraphs
from BigData.BigDataset import GetFeatures
from Data.Graphs import CreateCompletions
from GNN.Training import EGAT
import torch
import sys
import os

modelpath = "GNN/model/egat_f23_f1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GetAcceptability(model, apxpath, task, argID):
    model.eval()
    filename = os.path.splitext(os.path.basename(apxpath))[0]
    graph, num_nodes, certain_nodes, nodes_id, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(apxpath)
    len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts, f"cache/{filename}.apx")
    if len_def_atts_MIN == 0:
        return "ERROR : Zero attack in the minimal completion"
    features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx", )
    features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx", )
    node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
    with torch.no_grad():
        node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
        predicted = (torch.sigmoid(node_out) > 0.5).tolist()
    argID = nodes_id[argID]
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