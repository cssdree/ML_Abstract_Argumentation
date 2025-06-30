from BigData.BigDataset import CreateDGLGraphs
from BigData.BigDataset import GetFeatures
from Data.Graphs import CreateCompletions
from GNN.Training import EGAT
from pathlib import Path
import subprocess
import statistics
import torch
import time
import ast
import os

#IAF_root = "A-inc"
IAF_root = "B-inc"
#sem = "ST"
#sem = "PR"
sem = "GR"
modelroot = f"../GNN/models/egat_f23_f1_{sem}.pth"
taeydennae_root = "../taeydennae_linux_x86-64"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if IAF_root == "A-inc":
    errors = {"afinput_exp_cycles_indvary3_step8_batch_yyy07_15_arg_inc",
              "afinput_exp_cycles_indvary3_step8_batch_yyy07_15_inc",
              "massachusetts_srta_2014-11-13.gml.50_5_att_inc",
              "massachusetts_vineyardfastferry_2015-11-13.gml.50_15_att_inc",
              "massachusetts_vineyardfastferry_2015-11-13.gml.50_15_inc",
              "massachusetts_vineyardfastferry_2015-11-13.gml.50_20_att_inc",
              "massachusetts_vineyardfastferry_2015-11-13.gml.50_20_inc"}
if IAF_root == "B-inc":
    errors = {"massachusetts_blockislandferry_2015-11-13.gml.80_15_arg_inc",
              "massachusetts_blockislandferry_2015-11-13.gml.80_15_inc"}


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
    os.makedirs(f"{IAF_root}/GNN_labels/crashs", exist_ok=True)
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            argpath = f"{IAF_root}/{filename}.arg"
            labelpath = f"{IAF_root}/GNN_labels/{filename}_{sem}.txt"
            crashpath = f"{IAF_root}/GNN_labels/crashs/{filename}_crash.txt"
            if not os.path.exists(labelpath) and not os.path.exists(crashpath) and filename not in errors:
                print(filename)
                with open(argpath, "r", encoding="utf-8") as f:
                    arg = f.readline().strip()
                start_time = time.time()
                graph, num_nodes, certain_nodes, nodes_id, is_node_uncertain, def_args, inc_args, def_atts, inc_atts = CreateDGLGraphs(apxpath)
                len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts,f"cache/{filename}.apx")
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


def GlobalStatistics():
    VP = 0  #Vrai Positif : args acceptés qu'on a classé comme accepctés
    VN = 0  #Vrai Négatif : args rejetés qu'on a classé comme rejetés
    FP = 0  #Faux Positif : args rejetés qu'on a classé comme acceptés
    FN = 0  #Faux Négatif : args acceptés qu'on a classé comme rejetés
    taeydennae_time = 0
    GNN_time = 0
    taeydennae_median = []
    GNN_median = []
    graphs_predicted_taey = 4550 - sum(1 for f in Path(f"{IAF_root}/taeydennae_labels/timeouts-{sem}").iterdir() if f.is_file())
    graphs_predicted_gnn = 0
    for txtfile in os.listdir(f"{IAF_root}/GNN_labels"):
        if txtfile.endswith(f"{sem}.txt"):
            filename = "_".join(os.path.splitext(txtfile)[0].split("_")[:-1])
            graphs_predicted_gnn += 1
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
    print(f"Graphs predicted by Taeydennae (less than 10 sec) : {graphs_predicted_taey}/4550")
    print(f"Graphs predicted by the GNN (no crash or error) : {graphs_predicted_gnn}/4550")
    print("")
    print("Taeydennae average time :", taeydennae_time/graphs_predicted_taey)
    print("Taeydennae median time :", statistics.median(taeydennae_median))
    print("Taeydennae cumulative time :", taeydennae_time)
    print("GNN average time :", GNN_time/graphs_predicted_gnn)
    print("GNN median time :", statistics.median(GNN_median))
    print("GNN cumulative time :", GNN_time)
    print("")
    print("Accuracy :",(VP+VN)/(VP+VN+FP+FN))
    print("Precision :",VP/(VP+FP))
    print("Recall :",VP/(VP+FN))
    print("F1-score :",(2*VP)/(2*VP+FP+FN))
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
    #TestTaeydennae()
    #model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    #model.load_state_dict(torch.load(modelroot, map_location=device))
    #TestGNN(model)
    GlobalStatistics()
    DecisionProblemStatistics()