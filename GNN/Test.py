from GNN.iaf_egat_f23_f1 import CreateCompletions
from GNN.Dataset import CreateDGLGraphs
from Data.Labeling import CertainsArgs
from GNN.Dataset import GetFeatures
from GNN.Dataset import GetLabels
from GNN.Training import EGAT
import subprocess
import torch
import time
import ast
import os

IAF_root = "../Data/IAF_TestSet"
taeydennae_root = "../taeydennae_linux_x86-64"
#sem = "ST"
sem = "PR"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TimeWithTaeydennae():
    nb_requests = 0
    start_time = time.time()
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            for task in ["PCA", "NCA", "PSA", "NSA"]:
                filename = os.path.splitext(apxfile)[0]
                certain_args = CertainsArgs(f"{IAF_root}/{filename}.apx")
                for arg in certain_args:
                    prediction = subprocess.run([taeydennae_root, "-p", f"{task}-{sem}", "-f", f"{IAF_root}/{filename}.apx", "-a", str(arg)], capture_output=True, text=True)
                    nb_requests += 1
    end_time = time.time()
    print(f"{nb_requests} requests took", end_time-start_time, "seconds with Taeydennae")


def TimeWithGNN(model):
    os.makedirs("cache", exist_ok=True)
    os.makedirs(f"{IAF_root}/predictions", exist_ok=True)
    model.eval()
    nb_graphs = 0
    start_time = time.time()
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            predictionpath = f"{IAF_root}/predictions/{filename}_{sem}.txt"
            graph, num_nodes, certain_nodes, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(apxpath)
            len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts, f"cache/{filename}.apx")
            features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx",f"cache/{filename}_MAX.pt")
            features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx",f"cache/{filename}_MIN.pt")
            node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
            with torch.no_grad():
                node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
                predictions = (torch.sigmoid(node_out) > 0.5).int().tolist()
            with open(predictionpath, "w", encoding="utf-8") as f:
                f.write(f"{predictions}\n")
                nb_graphs += 1
    end_time = time.time()
    print(f"{nb_graphs} whole graphs took", end_time-start_time, "seconds with the GNN")


def Statistics():
    VP = 0  #Vrai Positif : args acceptés qu'on a classé comme accepctés
    VN = 0  #Vrai Négatif : args rejetés qu'on a classé comme rejetés
    FP = 0  #Faux Positif : args rejetés qu'on a classé comme acceptés
    FN = 0  #Faux Négatif : args acceptés qu'on a classé comme rejetés
    PCA = {"name": "PCA", "VP": 0, "VN": 0, "FP": 0, "FN": 0}
    NCA = {"name": "NCA", "VP": 0, "VN": 0, "FP": 0, "FN": 0}
    PSA = {"name": "PSA", "VP": 0, "VN": 0, "FP": 0, "FN": 0}
    NSA = {"name": "NSA", "VP": 0, "VN": 0, "FP": 0, "FN": 0}
    problems = [PCA, NCA, PSA, NSA]
    for txtfile in os.listdir(f"{IAF_root}/predictions"):
        if txtfile.endswith(".txt"):
            filename = "_".join(os.path.splitext(txtfile)[0].split("_")[:-1])
            csvpath = f"{IAF_root}/labels/{filename}_{sem}.csv"
            num_nodes = int(filename.split("_")[1])
            labels = GetLabels(num_nodes, csvpath, device).tolist()
            with open(f"{IAF_root}/predictions/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                predictions = ast.literal_eval(f.readline().strip())
            for l_list, p_list in zip(labels, predictions):
                problem_id = 0
                for l_val, p_val in zip(p_list, l_list):
                    problem_dict = problems[problem_id]
                    if p_val == l_val == 1:
                        problem_dict["VP"] += 1
                        VP += 1
                    elif p_val == l_val == 0:
                        problem_dict["VN"] += 1
                        VN += 1
                    elif p_val == 1 and l_val == 0:
                        problem_dict["FP"] += 1
                        FP += 1
                    elif p_val == 0 and l_val == 1:
                        problem_dict["FN"] += 1
                        FN += 1
                    problem_id += 1
    print("Accuracy :",(VP+VN)/(VP+VN+FP+FN))
    print("Precision :",VP/(VP+FP))
    print("Recall :",VP/(VP+FN))
    print("F1-score :",(2*VP)/(2*VP+FP+FN))
    print("")
    for problem in problems:
        problem_name = problem["name"]
        print(f"{problem_name} Accuracy :", (problem["VP"]+problem["VN"])/(problem["VP"]+problem["VN"]+problem["FP"]+problem["FN"]))
        print(f"{problem_name} Precision :", problem["VP"]/(problem["VP"]+problem["FP"]))
        print(f"{problem_name} Recall :", problem["VP"]/(problem["VP"]+problem["FN"]))
        print(f"{problem_name} F1-score :", (2*problem["VP"])/(2*problem["VP"]+problem["FP"]+problem["FN"]))
        print("")


if __name__ == "__main__":
    #TimeWithTaeydennae()
    #model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    #model.load_state_dict(torch.load(f"models/egat_f23_f1_{sem}.pth", map_location=device))
    #TimeWithGNN(model)
    Statistics()