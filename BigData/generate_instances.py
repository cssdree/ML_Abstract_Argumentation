import os
import sys
import numpy as np

# source folder for base AFs
original_instance_folder = sys.argv[1]
# output folder for incomplete AFs
generated_instance_folder = sys.argv[2]
# seed used for AAAI'20 paper
seed = 20190905
# probability of an element being uncertain
probs = [0, 0.05, 0.1, 0.15, 0.2]

np.random.seed(seed)

count = 0
for file in open(original_instance_folder + ".txt").read().split("\n"):
    count += 1
    if file.endswith(".apx"):
        lines = open(original_instance_folder + "/" + file, "r").read().split("\n")
        args = [line for line in lines if line.startswith("arg")]
        query = np.random.choice(args)
        q = query.replace("arg(", "").replace(").", "")
        atts = [line for line in lines if line.startswith("att")]
        for p in probs:
            if p == 0:
                filename = file.replace(".apx", "") + "_" + str(int(100 * p)) + "_inc.apx"
                out = open(generated_instance_folder + "/" + filename, "w")
                for arg in args:
                    out.write(arg + "\n")
                for att in atts:
                    out.write(att + "\n")
                out.close()
                qout = open(generated_instance_folder + "/" + filename.replace(".apx", ".arg"), "w")
                qout.write(q)
                qout.close()
                continue

            inc_args = []
            inc_atts = []
            def_args = []
            def_atts = []
            for arg in args:
                if np.random.uniform() < p and arg != query:
                    inc_args += [arg]
                else:
                    def_args += [arg]
            for att in atts:
                if np.random.uniform() < p:
                    inc_atts += [att]
                else:
                    def_atts += [att]

            filename = file.replace(".apx", "") + "_" + str(int(100 * p)) + "_inc.apx"
            out = open(generated_instance_folder + "/" + filename, "w")
            for arg in def_args:
                out.write(arg + "\n")
            for arg in inc_args:
                out.write("?" + arg + "\n")
            for att in def_atts:
                out.write(att + "\n")
            for att in inc_atts:
                out.write("?" + att + "\n")
            out.close()
            qout = open(generated_instance_folder + "/" + filename.replace(".apx", ".arg"), "w")
            qout.write(q)
            qout.close()

            filename = file.replace(".apx", "") + "_" + str(int(100 * p)) + "_arg_inc.apx"
            out = open(generated_instance_folder + "/" + filename, "w")
            for arg in def_args:
                out.write(arg + "\n")
            for arg in inc_args:
                out.write("?" + arg + "\n")
            for att in atts:
                out.write(att + "\n")
            out.close()
            qout = open(generated_instance_folder + "/" + filename.replace(".apx", ".arg"), "w")
            qout.write(q)
            qout.close()

            filename = file.replace(".apx", "") + "_" + str(int(100 * p)) + "_att_inc.apx"
            out = open(generated_instance_folder + "/" + filename, "w")
            for arg in args:
                out.write(arg + "\n")
            for att in def_atts:
                out.write(att + "\n")
            for att in inc_atts:
                out.write("?" + att + "\n")
            out.close()
            qout = open(generated_instance_folder + "/" + filename.replace(".apx", ".arg"), "w")
            qout.write(q)
            qout.close()