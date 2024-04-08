import os

cwd = os.path.dirname(os.path.abspath(__file__))
if cwd not in os.environ.keys():
    os.environ["cwd"] = cwd
with open(os.path.join(cwd, "Dataset_Location.txt"), "r") as f:
    lines = f.readlines()
lines = [line.replace("\n", "") for line in lines]
dataset_outer_fp = lines[0]
frames_outer_fp = lines[1]
