import concurrent.futures
import subprocess
import csv
import os

IAF_root = "IAF_TrainSet"
#IAF_root = "IAF_TestSet"
labels_root = f"{IAF_root}/labels"
taeydennae_root = "../taeydennae_linux_x86-64"
decision_problems = ["PCA", "NCA", "PSA", "NSA"]
semantics = ["ST"]
graphs_results = {}  #dict that contains the acceptability of each arg for each decision problem


def Task(filename, sem, arg, problem):
    result = subprocess.run(
        [taeydennae_root, "-p", f"{problem}-{sem}", "-f", f"{IAF_root}/{filename}.apx", "-a", str(arg)],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and "YES" in result.stdout:
        return filename, sem, arg, problem, 1
    elif result.returncode == 0 and "NO" in result.stdout:
        return filename, sem, arg, problem, 0
    else:
        print(f"Error : {filename}, problem={problem}, sem={sem}, arg={arg}, result={result.stdout}")
        return filename, sem, arg, problem, 2


def CertainsArgs(file):
    """
    Return all the arguments of a file that are not uncertains
    """
    certain_args = set()
    with open(f"{IAF_root}/{file}", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('arg(') and line.endswith(').'):
                certain_args.add(line[4:-2])
    return certain_args


#Filling the "graphs_results" dictionary
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = []
    for sem in semantics:
        graphs_results[sem] = {}
        for file in os.listdir(IAF_root):
            if not file.endswith(".apx"):
                continue
            filename = os.path.splitext(file)[0]
            if filename not in graphs_results[sem]:
                graphs_results[sem][filename] = {}
            certain_args = CertainsArgs(file)
            for arg in certain_args:
                if arg not in graphs_results[sem][filename]:
                    graphs_results[sem][filename][arg] = {}
                for problem in decision_problems:
                    futures.append(executor.submit(Task, filename, sem, arg, problem))
    for future in concurrent.futures.as_completed(futures):
        filename, sem, arg, problem, result = future.result()
        graphs_results[sem][filename][arg][problem] = result

#Building the csv files from "graphs_results" filled in
os.makedirs(labels_root, exist_ok=True)
for sem in graphs_results.keys():
    for file in graphs_results[sem].keys():
        csvpath = f"{labels_root}/{file}_{sem}.csv"
        with open(csvpath, "w", newline="", encoding="utf-8") as f:
            f.write("#Argument-PCA-NCA-PSA-NSA\n")
            for arg in graphs_results[sem][file].keys():
                writer = csv.writer(f)
                writer.writerow(
                    [arg,
                     graphs_results[sem][file][arg]["PCA"],
                     graphs_results[sem][file][arg]["NCA"],
                     graphs_results[sem][file][arg]["PSA"],
                     graphs_results[sem][file][arg]["NSA"]
                     ]
                )