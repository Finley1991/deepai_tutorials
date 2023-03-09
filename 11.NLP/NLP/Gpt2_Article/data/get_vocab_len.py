import torch
import os
DIR_PATH = r"../data/books_utf_8"

for i, filename in enumerate(os.listdir(DIR_PATH)):
    f_path = os.path.join(DIR_PATH, filename)
    with open(f_path, "r+") as f:
        print(len(f.read()))
line = open(os.path.join("../data/vocab.txt"), "r+", encoding="utf-8").read()
print(len(line.split()))
