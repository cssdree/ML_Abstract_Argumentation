import os
import sys
import numpy as np


original_instance_folder = sys.argv[1] #source folder for AFs
generated_instance_folder = sys.argv[2] #output folder for IAFs
probs = [0, 0.05, 0.1, 0.15, 0.2] #probabilities that an argument is uncertain

seed = 12122025
np.random.seed(seed)

count = 0
for file in os.listdir(original_instance_folder):
	count += 1
	if file.endswith(".apx"):
		arg_filename = file.replace(".apx", ".arg")
		arg_filepath = os.path.join(original_instance_folder, arg_filename)
		try:
			with open(arg_filepath, "r") as qin:
				q = qin.read().strip()
		except FileNotFoundError:
			print(f"WARNING: Skipping {file} - associated .arg file not found.")
			continue
		query = f"arg({q})."
		with open(os.path.join(original_instance_folder, file), "r") as f:
			lines = f.read().split("\n")
		args = [line for line in lines if line.startswith("arg")]
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

			filename = file.replace(".apx","") + "_" + str(int(100*p)) + "_inc.apx"
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
			qout = open(generated_instance_folder + "/" + filename.replace(".apx",".arg"), "w")
			qout.write(q)
			qout.close()

			filename = file.replace(".apx","") + "_" + str(int(100*p)) + "_arg_inc.apx"
			out = open(generated_instance_folder + "/" + filename, "w")
			for arg in def_args:
				out.write(arg + "\n")
			for arg in inc_args:
				out.write("?" + arg + "\n")
			for att in atts:
				out.write(att + "\n")
			out.close()
			qout = open(generated_instance_folder + "/" + filename.replace(".apx",".arg"), "w")
			qout.write(q)
			qout.close()

			filename = file.replace(".apx","") + "_" + str(int(100*p)) + "_att_inc.apx"
			out = open(generated_instance_folder + "/" + filename, "w")
			for arg in args:
				out.write(arg + "\n")
			for att in def_atts:
				out.write(att + "\n")
			for att in inc_atts:
				out.write("?" + att + "\n")
			out.close()
			qout = open(generated_instance_folder + "/" + filename.replace(".apx",".arg"), "w")
			qout.write(q)
			qout.close()