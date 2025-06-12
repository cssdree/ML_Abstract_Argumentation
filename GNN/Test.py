from GNN.iaf_egat_f23_f1 import CreateCompletions
from GNN.Dataset import CreateDGLGraphs
from Data.Labeling import CertainsArgs
from GNN.Dataset import GetFeatures
from GNN.Training import EGAT
import subprocess
import Dataset
import torch
import time
import os

device = "cpu"
cache_root = "../cache"
IAF_root = "../Data/IAF_TestSet"
modelpath = "model/egat_f23_f1.pth"
taeydennae_root = "../taeydennae_linux_x86-64"


def Statistics(model, device="cpu"):
    actual_yes = {"actual_PCA_yes":0, "actual_NCA_yes":0, "actual_PSA_yes":0, "actual_NSA_yes":0}
    actual_no = {"actual_PCA_no": 0, "actual_NCA_no": 0, "actual_PSA_no": 0,"actual_NSA_no": 0}
    correct_yes = {"correct_PCA_yes":0, "correct_NCA_yes":0, "correct_PSA_yes":0, "correct_NSA_yes":0}
    correct_no = {"correct_PCA_no":0, "correct_NCA_no":0, "correct_PSA_no":0, "correct_NSA_no":0}
    total_actual_yes_no = 0
    total_correct_yes_no = 0
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
            for task_idx, task_name in enumerate(["PCA", "NCA", "PSA", "NSA"]):
                predicted_task = [p[task_idx] for p in predicted]
                label_task = [l[task_idx] for l in label]
                one_actual_yes = sum(l == 1.0 for l in label_task)
                one_actual_no = sum(l == 0.0 for l in label_task)
                one_correct_yes = sum(p == l == 1.0 for p, l in zip(predicted_task, label_task))
                one_correct_no = sum(p == l == 0.0 for p, l in zip(predicted_task, label_task))
                actual_yes[f"actual_{task_name}_yes"] += one_actual_yes
                actual_no[f"actual_{task_name}_no"] += one_actual_no
                correct_yes[f"correct_{task_name}_yes"] += one_correct_yes
                correct_no[f"correct_{task_name}_no"] += one_correct_no
    print("Pourcentages de prédictions correctes :")
    for task_name in ["PCA", "NCA", "PSA", "NSA"]:
        print(f"Yes {task_name} :",(correct_yes[f"correct_{task_name}_yes"]/actual_yes[f"actual_{task_name}_yes"])*100,
              f"No {task_name} :",(correct_no[f"correct_{task_name}_no"]/actual_no[f"actual_{task_name}_no"])*100,
              f"Total {task_name} :",((correct_yes[f"correct_{task_name}_yes"]+correct_no[f"correct_{task_name}_no"])/(actual_yes[f"actual_{task_name}_yes"]+actual_no[f"actual_{task_name}_no"]))*100)
        total_actual_yes_no += (actual_yes[f"actual_{task_name}_yes"]+actual_no[f"actual_{task_name}_no"])
        total_correct_yes_no += (correct_yes[f"correct_{task_name}_yes"]+correct_no[f"correct_{task_name}_no"])
    print("TOTAL :",(total_correct_yes_no/total_actual_yes_no)*100,"% des prédictions sont correctes")

"""
def TimeWithGNN(model):
    start_time = time.time()
    for file in sorted(os.listdir(IAF_root))[:100]: #Traitement des 50 premiers fichiers apx
        if file.endswith(".apx"):
            for task in ["PCA", "NCA", "PSA", "NSA"]:
                filename = os.path.splitext(file)[0]
                certain_args = CertainsArgs(IAF_root, f"{filename}.apx")
                for arg in certain_args:
                    GetAcceptability(model, cache_root, f"{IAF_root}/{filename}",task,str(arg))
    end_time = time.time()
    return (end_time-start_time)
"""

def TimeWithGNN(model):
    start_time = time.time()
    for i, file in enumerate(sorted(os.listdir(IAF_root))[:300], start=1):
        if file.endswith(".apx"):
            filename = os.path.splitext(file)[0]
            certain_args = CertainsArgs(IAF_root, f"{filename}.apx")
            filepath = f"{IAF_root}/{filename}"

            # Création du graphe et des features UNE SEULE FOIS
            graph, num_nodes, certain_nodes, is_node_uncertain = CreateDGLGraphs(f"{filepath}.apx")
            features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx", cache_root, "MAX"), f"{cache_root}/{filename}_MAX.pt")
            features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(f"{filepath}.apx", cache_root, "MIN"), f"{cache_root}/{filename}_MIN.pt")
            node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)

            with torch.no_grad():
                node_out, _ = model(graph, node_feats, graph.edata["is_uncertain"])
                predicted = (torch.sigmoid(node_out) > 0.5).tolist()

            # Boucle sur les prédictions déjà faites
            for task_idx, task in enumerate(["PCA", "NCA", "PSA", "NSA"]):
                for arg in certain_args:
                    prediction = predicted[int(arg)][task_idx]
                    # Tu peux stocker ou afficher ici
                    _ = "YES" if prediction else "NO"
    end_time = time.time()
    return end_time - start_time


def TimeWithTaeydennae():
    total_requests = 0
    start_time = time.time()
    for file in sorted(os.listdir(IAF_root))[:300]: #Traitement des 50 premiers fichiers apx
        if file.endswith(".apx"):
            for task in ["PCA", "NCA", "PSA", "NSA"]:
                filename = os.path.splitext(file)[0]
                certain_args = CertainsArgs(IAF_root, f"{filename}.apx")
                for arg in certain_args:
                    subprocess.run([taeydennae_root, "-p", f"{task}-ST", "-f", f"{IAF_root}/{filename}.apx", "-a", str(arg)], capture_output=True, text=True)
                    total_requests += 1
    end_time = time.time()
    return (end_time-start_time), total_requests


os.makedirs(cache_root, exist_ok=True)
model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
model.load_state_dict(torch.load(modelpath, map_location=device))
#Statistics(model)
time_GNN = TimeWithGNN(model)
time_taeydennae, total_requests = TimeWithTaeydennae()
print("Time with the GNN :", time_GNN)
print("Time with taeydennae :", time_taeydennae)
print(total_requests, "requests have been made")