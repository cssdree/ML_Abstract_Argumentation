import concurrent.futures
import subprocess
import csv
import os

#IAF_root = "IAF_TrainSet"
IAF_root = "IAF_TestSet"
taeydennae_root = "../taeydennae_linux_x86-64"


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


def CertainsArgs(apxpath):
    """
    Return all the arguments of a file that are not uncertains
    """
    certain_args = set()
    with open(apxpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('arg(') and line.endswith(').'):
                certain_args.add(line[4:-2])
    return certain_args


if __name__ == "__main__":
    #Filling the "graphs_results" dictionary
    graphs_results = {}
    semantics = ["ST", "PR", "GR"]
    decision_problems = ["PCA", "NCA", "PSA", "NSA"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for sem in semantics:
            graphs_results[sem] = {}
            for apxfile in os.listdir(IAF_root):
                if apxfile.endswith(".apx"):
                    filename = os.path.splitext(apxfile)[0]
                    if filename not in graphs_results[sem]:
                        graphs_results[sem][filename] = {}
                    certain_args = CertainsArgs(f"{IAF_root}/{filename}.apx")
                    for arg in certain_args:
                        if arg not in graphs_results[sem][filename]:
                            graphs_results[sem][filename][arg] = {}
                        for problem in decision_problems:
                            futures.append(executor.submit(Task, filename, sem, arg, problem))
        for future in concurrent.futures.as_completed(futures):
            filename, sem, arg, problem, result = future.result()
            graphs_results[sem][filename][arg][problem] = result

    #Building the csv files from "graphs_results" filled in
    os.makedirs(f"{IAF_root}/labels", exist_ok=True)
    for sem in graphs_results.keys():
        for file in graphs_results[sem].keys():
            csvpath = f"{IAF_root}/labels/{file}_{sem}.csv"
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