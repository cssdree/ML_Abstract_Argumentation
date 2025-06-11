from GNN.Dataset import Dataset
from GNN.Training import EGAT
import torch

device = "cpu"
IAF_root = "../Data/IAF_TestSet"
modelpath = "model/egat_f23_f1.pth"


def Statistics(model, device="cpu"):
    total_actual_yes = 0
    correct_yes_predictions = 0
    total_actual_no = 0
    correct_no_predictions = 0
    iaf_dataset = Dataset(IAF_root)
    model.eval()
    with torch.no_grad():
        for graph in iaf_dataset:
            node_feats = graph.ndata["feat"].to(device)
            edge_feats = graph.edata["is_uncertain"].to(device)
            label = graph.ndata["label"].to(device)
            mask = graph.ndata["mask"].to(device).bool()
            node_out, edge_out = model(graph, node_feats, edge_feats)
            predicted = (torch.sigmoid(node_out[mask])>0.5).float().tolist()
            label = label[mask].tolist()
            one_prediction_of_yes = sum(p == l == 1.0 for p, l in zip(predicted, label))  #marche pas car p et l sont des listes
            one_prediction_of_no = sum(p == l == 0.0 for p, l in zip(predicted, label))  #marche pas car p et l sont des listes
            correct_yes_predictions += one_prediction_of_yes
            correct_no_predictions += one_prediction_of_no
            actual_yes = sum(l == 1.0 for l in label)  #marche pas car l est une liste
            actual_no = sum(l == 0.0 for l in label)  #marche pas car l est une liste
            total_actual_yes += actual_yes
            total_actual_no += actual_no
    print("Pourcentage de prédictions correctes : ",(correct_yes_predictions+correct_no_predictions)/(total_actual_yes+total_actual_no)*100)
    print("Pourcentage de YES corrects : ",(correct_yes_predictions/total_actual_yes)*100)
    print("Pourcentage de NO corrects : ", (correct_no_predictions / total_actual_no)*100)
    #DIFFERENCIER EN FONCTION DES PROBLEMES DE DECISION

def TimeWithGNN():
    return None

def TimeWithTaeydennae():
    return None


model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
model.load_state_dict(torch.load(modelpath, map_location=device))
Statistics(model)