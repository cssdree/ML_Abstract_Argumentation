from BigData.BigDataset import CreateDGLGraphs
from BigData.BigDataset import GetFeatures
from Data.Graphs import CreateCompletions
from GNN.Training import EGAT
import subprocess
import statistics
import torch
import time
import ast
import os

taeydennae_root = "./taeydennae_linux_x86-64"
IAF_root = "BigData/A-inc"
#IAF_root = "B-inc"
#sem = "ST"
#sem = "PR"
sem = "GR"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TestTaeydennae():
    os.makedirs(f"{IAF_root}/taeydennae_labels", exist_ok=True)
    os.makedirs(f"{IAF_root}/taeydennae_labels/timeouts-{sem}", exist_ok=True)
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            argpath = f"{IAF_root}/{filename}.arg"
            labelpath = f"{IAF_root}/taeydennae_labels/{filename}_{sem}.txt"
            timeoutpath = f"{IAF_root}/taeydennae_labels/timeouts-{sem}/{filename}_timeout.txt"
            if not os.path.exists(labelpath) and not os.path.exists(timeoutpath):
                print(filename)
                predictions = []
                timeout_occurred = False
                with open(argpath, "r", encoding="utf-8") as f:
                    arg = f.readline().strip()
                start_time = time.time()
                for task in ["PCA", "NCA", "PSA", "NSA"]:
                    try:
                        prediction = subprocess.run([taeydennae_root, "-p", f"{task}-{sem}", "-f", apxpath, "-a", str(arg)], capture_output=True, text=True, timeout=10)
                        if prediction.returncode == 0 and "YES" in prediction.stdout:
                            predictions.append(1)
                        elif prediction.returncode == 0 and "NO" in prediction.stdout:
                            predictions.append(0)
                        else:
                            print(f"Error : {filename}, task={task}, arg={arg}, prediction={prediction.stdout}")
                            predictions.append(2)
                    except subprocess.TimeoutExpired:
                        open(timeoutpath, "w").close()
                        timeout_occurred = True
                        break
                end_time = time.time()
                predictions_time = end_time - start_time
                if not timeout_occurred:
                    with open(labelpath, "w", encoding="utf-8") as f:
                        f.write(f"{predictions}\n")
                        f.write(f"{predictions_time}\n")


def TestGNN(model):
    os.makedirs("cache", exist_ok=True)
    os.makedirs(f"{IAF_root}/GNN_labels", exist_ok=True)
    os.makedirs(f"{IAF_root}/GNN_labels/failed-{sem}", exist_ok=True)
    os.makedirs(f"{IAF_root}/GNN_labels/failed-{sem}/crash", exist_ok=True)
    os.makedirs(f"{IAF_root}/GNN_labels/failed-{sem}/error", exist_ok=True)
    for apxfile in os.listdir(f"{IAF_root}"):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            argpath = f"{IAF_root}/{filename}.arg"
            labelpath = f"{IAF_root}/GNN_labels/{filename}_{sem}.txt"
            crashpath = f"{IAF_root}/GNN_labels/failed-{sem}/crash/{filename}_crash.txt"
            errorpath = f"{IAF_root}/GNN_labels/failed-{sem}/error/{filename}_error.txt"
            if not os.path.exists(labelpath) and not os.path.exists(crashpath) and not os.path.exists(errorpath):
                print(filename)
                with open(argpath, "r", encoding="utf-8") as f:
                    arg = f.readline().strip()
                start_time = time.time()
                graph, num_nodes, certain_nodes, nodes_id, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(apxpath)
                len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts,f"cache/{filename}.apx")
                if len_def_atts_MIN == 0:
                    open(errorpath, "w").close()
                    continue
                features_MAX = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MAX.apx")
                features_MIN = GetFeatures(num_nodes, certain_nodes, f"cache/{filename}_MIN.apx")
                node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
                with torch.no_grad():
                    node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
                    predictions = (torch.sigmoid(node_out) > 0.5).int().tolist()
                    prediction = predictions[nodes_id[str(arg)]]
                end_time = time.time()
                predictions_time = end_time - start_time
                with open(labelpath, "w", encoding="utf-8") as f:
                    f.write(f"{prediction}\n")
                    f.write(f"{predictions_time}\n")


def CountFiles(root):
    count = 0
    if root == f"{IAF_root}/GNN_labels" or root == f"{IAF_root}/taeydennae_labels":
        for file in os.listdir(root):
            if file.endswith(f"{sem}.txt"):
                count += 1
        return count
    if root == f"{IAF_root}/GNN_labels/failed-{sem}":
        for file in os.listdir(f"{root}/crash"):
            count += 1
        for file in os.listdir(f"{root}/error"):
            count += 1
        return count
    if root == f"{IAF_root}/taeydennae_labels/timeouts-{sem}":
        for file in os.listdir(root):
            count += 1
        return count


def GlobalStatistics():
    VP = 0  #Vrai Positif : args acceptés qu'on a classé comme accepctés
    VN = 0  #Vrai Négatif : args rejetés qu'on a classé comme rejetés
    FP = 0  #Faux Positif : args rejetés qu'on a classé comme acceptés
    FN = 0  #Faux Négatif : args acceptés qu'on a classé comme rejetés
    taeydennae_time = 0
    GNN_time = 0
    taeydennae_median = []
    GNN_median = []
    nb_graphs_predicted_GNN = CountFiles(f"{IAF_root}/GNN_labels")
    nb_graphs_predicted_taey = CountFiles(f"{IAF_root}/taeydennae_labels")
    nb_graphs_failed_GNN = CountFiles(f"{IAF_root}/GNN_labels/failed-{sem}")
    nb_graphs_failed_taey = CountFiles(f"{IAF_root}/taeydennae_labels/timeouts-{sem}")
    for txtfile in os.listdir(f"{IAF_root}/GNN_labels"):
        if txtfile.endswith(f"{sem}.txt"):
            filename = "_".join(os.path.splitext(txtfile)[0].split("_")[:-1])
            if os.path.exists(f"{IAF_root}/taeydennae_labels/timeouts-{sem}/{filename}_timeout.txt"):
                taeydennae_prediction_time = 20
                taeydennae_median.append(taeydennae_prediction_time)
                with open(f"{IAF_root}/GNN_labels/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                    GNN_prediction = ast.literal_eval(f.readline().strip())
                    GNN_prediction_time = float(f.readline().strip())
                    GNN_median.append(GNN_prediction_time)
            else :
                with open(f"{IAF_root}/taeydennae_labels/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                    taeydennae_prediction = ast.literal_eval(f.readline().strip())
                    taeydennae_prediction_time = float(f.readline().strip())
                    taeydennae_median.append(taeydennae_prediction_time)
                with open(f"{IAF_root}/GNN_labels/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                    GNN_prediction = ast.literal_eval(f.readline().strip())
                    GNN_prediction_time = float(f.readline().strip())
                    GNN_median.append(GNN_prediction_time)
                for gnn_val, taey_val in zip(GNN_prediction, taeydennae_prediction):
                    if gnn_val == taey_val == 1:
                        VP += 1
                    elif gnn_val == taey_val == 0:
                        VN += 1
                    elif gnn_val == 1 and taey_val == 0:
                        FP += 1
                    elif gnn_val == 0 and taey_val == 1:
                        FN += 1
            taeydennae_time += taeydennae_prediction_time
            GNN_time += GNN_prediction_time
    print("Accuracy :",(VP+VN)/(VP+VN+FP+FN))
    print("Precision :",VP/(VP+FP))
    print("Recall :",VP/(VP+FN))
    print("F1-score :",(2*VP)/(2*VP+FP+FN))
    print("")
    print("Taeydennae average time :", taeydennae_time/nb_graphs_predicted_taey)
    print("Taeydennae median time :", statistics.median(taeydennae_median))
    print("Taeydennae cumulative time :", taeydennae_time)
    print("GNN average time :", GNN_time/nb_graphs_predicted_GNN)
    print("GNN median time :", statistics.median(GNN_median))
    print("GNN cumulative time :", GNN_time)
    print("")
    print("Graphs correctly predicted by the GNN :", nb_graphs_predicted_GNN)
    print("Graphs failed by the GNN (error or crash) :", nb_graphs_failed_GNN)
    print("Graphs correctly predicted by Taeydennae :", nb_graphs_predicted_taey)
    print("Graphs failed by Taeydennae (timeout of 10 sec) :", nb_graphs_failed_taey)
    print("")


def DecisionProblemStatistics():
    PCA = {"name": "PCA", "VP": 0, "VN": 0, "FP": 0, "FN":0}
    NCA = {"name": "NCA", "VP": 0, "VN": 0, "FP": 0, "FN":0}
    PSA = {"name": "PSA", "VP": 0, "VN": 0, "FP": 0, "FN":0}
    NSA = {"name": "NSA", "VP": 0, "VN": 0, "FP": 0, "FN":0}
    problems = [PCA, NCA, PSA, NSA]
    for txtfile in os.listdir(f"{IAF_root}/GNN_labels"):
        if txtfile.endswith(f"{sem}.txt"):
            filename = "_".join(os.path.splitext(txtfile)[0].split("_")[:-1])
            if not os.path.exists(f"{IAF_root}/taeydennae_labels/timeouts-{sem}/{filename}_timeout.txt"):
                with open(f"{IAF_root}/taeydennae_labels/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                    taeydennae_prediction = ast.literal_eval(f.readline().strip())
                with open(f"{IAF_root}/GNN_labels/{filename}_{sem}.txt", "r", encoding="utf-8") as f:
                    GNN_prediction = ast.literal_eval(f.readline().strip())
                problem_id = 0
                for gnn_val, taey_val in zip(GNN_prediction, taeydennae_prediction):
                    problem_dict = problems[problem_id]
                    if gnn_val == taey_val == 1:
                        problem_dict["VP"] += 1
                    elif gnn_val == taey_val == 0:
                        problem_dict["VN"] += 1
                    elif gnn_val == 1 and taey_val == 0:
                        problem_dict["FP"] += 1
                    elif gnn_val == 0 and taey_val == 1:
                        problem_dict["FN"] += 1
                    problem_id += 1
    for problem in problems:
        problem_name = problem["name"]
        print(f"{problem_name} Accuracy :", (problem["VP"]+problem["VN"])/(problem["VP"]+problem["VN"]+problem["FP"]+problem["FN"]))
        print(f"{problem_name} Precision :", problem["VP"]/(problem["VP"]+problem["FP"]))
        print(f"{problem_name} Recall :", problem["VP"]/(problem["VP"]+problem["FN"]))
        print(f"{problem_name} F1-score :", (2*problem["VP"])/(2*problem["VP"]+problem["FP"]+problem["FN"]))
        print("")


if __name__ == "__main__":
    TestTaeydennae()
    #model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    #model.load_state_dict(torch.load(f"../GNN/models/egat_f23_f1_{sem}.pth", map_location=device))
    #TestGNN(model)
    #GlobalStatistics()
    #DecisionProblemStatistics()