from BigData.Big_iaf_egat_f23_f1 import CreateCompletions
from BigData.BigDataset import CreateDGLGraphs
from BigData.BigDataset import GetFeatures
from GNN.Training import EGAT
import subprocess
import torch
import time
import os

IAF_root = "A-inc"
modelpath = "../GNN/model/egat_f23_f1.pth"
taeydennae_root = "../taeydennae_linux_x86-64"
device = "cpu"


def TestTaeydennae():
    os.makedirs(f"{IAF_root}/labels", exist_ok=True)
    os.makedirs(f"{IAF_root}/timeouts", exist_ok=True)
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            argpath = f"{IAF_root}/{filename}.arg"
            labelpath = f"{IAF_root}/labels/{filename}_ST.txt"
            timeoutpath = f"{IAF_root}/timeouts/{filename}_timeout.txt"
            if not os.path.exists(labelpath):
                predictions = []
                timeout_occurred = False
                with open(argpath, "r", encoding="utf-8") as f:
                    arg = f.readline().strip()
                start_time = time.time()
                for task in ["PCA", "NCA", "PSA", "NSA"]:
                    try:
                        prediction = subprocess.run([taeydennae_root, "-p", f"{task}-ST", "-f", apxpath, "-a", str(arg)], capture_output=True, text=True, timeout=2)
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
    os.makedirs(f"{IAF_root}/times", exist_ok=True)
    for apxfile in os.listdir(IAF_root):
        if apxfile.endswith(".apx"):
            filename = os.path.splitext(apxfile)[0]
            apxpath = f"{IAF_root}/{filename}.apx"
            argpath = f"{IAF_root}/{filename}.arg"
            timepath = f"{IAF_root}/{filename}.txt"
            if os.path.exists(f"{IAF_root}/labels/{filename}_ST.txt"):
                with open(argpath, "r", encoding="utf-8") as f:
                    arg = f.readline().strip()
                start_time = time.time()
                graph, num_nodes, certain_nodes, nodes_id, is_node_uncertain = CreateDGLGraphs(apxpath)
                features_MAX = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MAX"),f"cache/{filename}_MAX.pt")
                features_MIN = GetFeatures(num_nodes, certain_nodes, CreateCompletions(apxpath, "MIN"),f"cache/{filename}_MIN.pt")
                node_feats = torch.cat([is_node_uncertain.unsqueeze(1), features_MAX, features_MIN], dim=1)
                with torch.no_grad():
                    node_out, edge_out = model(graph, node_feats, graph.edata["is_uncertain"])
                    predictions = (torch.sigmoid(node_out) > 0.5).int().tolist()
                    prediction = predictions[nodes_id[str(arg)]]
                end_time = time.time()
                predictions_time = end_time - start_time
                print(prediction, predictions_time)

if __name__ == "__main__":
    #TestTaeydennae()
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    TestGNN(model)