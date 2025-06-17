from Data.Graphs import CreateCompletions
from GNN.Dataset import CreateDGLGraphs
from Data.Labeling import CertainsArgs
from GNN.Dataset import GetFeatures
from GNN.Dataset import Dataset
from GNN.Training import EGAT
import subprocess
import torch
import time
import os

IAF_root = "../Data/IAF_TestSet"
modelpath = "model/egat_f23_f1.pth"
taeydennae_root = "../taeydennae_linux_x86-64"
device = "cpu"


def Statistics(model, device="cpu"):
    VP = 0  #Vrai Positif : args acceptés qu'on a classé comme accepctés
    VN = 0  #Vrai Négatif : args rejetés qu'on a classé comme rejetés
    FP = 0  #Faux Positif : args rejetés qu'on a classé comme acceptés
    FN = 0  #Faux Négatif : args acceptés qu'on a classé comme rejetés
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
            for p_list, l_list in zip(predicted, label):
                for p_val, l_val in zip(p_list, l_list):
                    if p_val == l_val == 1:
                        VP += 1
                    elif p_val == l_val == 0:
                        VN += 1
                    elif p_val == 1 and l_val == 0:
                        FP += 1
                    elif p_val == 0 and l_val == 1:
                        FN += 1
    print("Accuracy :",(VP+VN)/(VP+VN+FP+FN))
    print("Precision :",VP/(VP+FP))
    print("Recall :",VP/(VP+FN))
    print("F1-score :",(2*VP)/(2*VP+FP+FN))


def TimeWithGNN(model):
    start_time = time.time()
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            graph, num_nodes, certain_nodes, is_node_uncertain = CreateDGLGraphs(apxpath)

            features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MAX"),f"cache/{filename}_MAX.pt")
            features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MIN"),f"cache/{filename}_MIN.pt")
            node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
            with torch.no_grad():
                node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
                predictions = (torch.sigmoid(node_out) > 0.5).tolist()
            for arg in certain_nodes:
                for task_idx, task_name in enumerate(["PCA", "NCA", "PSA", "NSA"]):
                    prediction = predictions[int(arg)][task_idx]
    end_time = time.time()
    return (end_time-start_time)


def TimeWithTaeydennae():
    total_requests = 0
    start_time = time.time()
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            for task in ["PCA", "NCA", "PSA", "NSA"]:
                filename = os.path.splitext(apxfile)[0]
                certain_args = CertainsArgs(f"{IAF_root}/{filename}.apx")
                for arg in certain_args:
                    prediction = subprocess.run([taeydennae_root, "-p", f"{task}-ST", "-f", f"{IAF_root}/{filename}.apx", "-a", str(arg)], capture_output=True, text=True)
                    total_requests += 1
    end_time = time.time()
    return (end_time-start_time), total_requests


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    Statistics(model)
    time_GNN = TimeWithGNN(model)
    time_taeydennae, total_requests = TimeWithTaeydennae()
    print("Time with the GNN :", time_GNN)
    print("Time with taeydennae :", time_taeydennae)
    print(total_requests, "requests have been made")