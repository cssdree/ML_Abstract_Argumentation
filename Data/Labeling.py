import concurrent.futures
import subprocess
import csv
import os

IAF_root = "Data/IAF_TrainSet"
#IAF_root = "Data/IAF_TestSet"
#sem = "ST"
sem = "PR"
#sem = "GR"
taeydennae_root = "./taeydennae_linux_x86-64"


def Task(filename, sem, arg, problem):
    result = subprocess.run(
        [taeydennae_root, "-p", f"{problem}-{sem}", "-f", f"{IAF_root}/{filename}.apx", "-a", str(arg)],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and "YES" in result.stdout:
        return filename, arg, problem, 1
    elif result.returncode == 0 and "NO" in result.stdout:
        return filename, arg, problem, 0
    else:
        print(f"Error : {filename}, problem={problem}, sem={sem}, arg={arg}, result={result.stdout}")
        return filename, arg, problem, 2


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for apxfile in os.listdir(IAF_root):
            if apxfile.endswith(".apx"):
                filename = os.path.splitext(apxfile)[0]
                graphs_results[filename] = {}
                certain_args = CertainsArgs(f"{IAF_root}/{filename}.apx")
                for arg in certain_args:
                    graphs_results[filename][arg] = {}
                    decision_problems = ["PCA", "NCA", "PSA", "NSA"]
                    for problem in decision_problems:
                        futures.append(executor.submit(Task, filename, sem, arg, problem))
        for future in concurrent.futures.as_completed(futures):
            filename, arg, problem, result = future.result()
            graphs_results[filename][arg][problem] = result
    #Building the csv files from "graphs_results" filled in
    os.makedirs(f"{IAF_root}/labels", exist_ok=True)
    for file in graphs_results.keys():
        csvpath = f"{IAF_root}/labels/{file}_{sem}.csv"
        with open(csvpath, "w", newline="", encoding="utf-8") as f:
            f.write("#Argument-PCA-NCA-PSA-NSA\n")
            for arg in graphs_results[file].keys():
                writer = csv.writer(f)
                writer.writerow(
                    [arg,
                     graphs_results[file][arg]["PCA"],
                     graphs_results[file][arg]["NCA"],
                     graphs_results[file][arg]["PSA"],
                     graphs_results[file][arg]["NSA"]
                     ]
                )