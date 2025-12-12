import sys

if len(sys.argv) != 2:
    print("ERROR: Syntax must be python3 af_to_apx.py file.af")
    sys.exit()

filename = sys.argv[1]

with open(filename, 'r') as file:
    lines = file.readlines()

arguments = []
attacks = []

for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("p"):
        splitting = line.split()
        if splitting[1] != "af":
            print("ERROR: The file is not the specification of an AF")
            sys.exit()
        nb_args = int(splitting[2])
        arguments = [i+1 for i in range(nb_args)]
    else:
        if len(arguments) == 0:
            print("ERROR: Missing specification of the number of arguments")
            sys.exit()
        else:
            splitting = line.split()
            if len(splitting) != 2:
                print(f"WARNING: Skipping unrecognised line in {filename}: {line}", file=sys.stderr)
                continue
            attacks.append([int(splitting[0]),int(splitting[1])])

for argument in arguments:
    print(f"arg({argument}).")

for attack in attacks:
    print(f"att({attack[0]},{attack[1]}).")