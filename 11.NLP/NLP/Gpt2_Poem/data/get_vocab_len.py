import torch
import os
DIR_PATH = r"books_convert"

for i, filename in enumerate(os.listdir(DIR_PATH)):
    f_path = os.path.join(DIR_PATH, filename)
    with open(f_path, "r+") as f:
        # print(f.read())
        #文章共多少字
        print(len(f.read()))
line = open(os.path.join("../data/vocab.txt"), "r+", encoding="utf-8").read()
# print(line.split())
#去重后剩余多少字
print(len(line.split()))
