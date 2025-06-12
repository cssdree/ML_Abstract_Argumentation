from GNN.Dataset import Dataset
from GNN.Training import EGAT
import torch

device = "cpu"
IAF_root = "../Data/IAF_TestSet"
modelpath = "model/egat_f23_f1.pth"


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


def TimeWithGNN():
    return None


def TimeWithTaeydennae():
    return None


model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
model.load_state_dict(torch.load(modelpath, map_location=device))
Statistics(model)