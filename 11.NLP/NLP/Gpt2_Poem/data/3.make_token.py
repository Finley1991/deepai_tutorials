import os

books_utf_8 = r"./books_convert"
books_tokenized = r"./books_tokenized"
vocab_path = "./vocab.txt"

if not os.path.exists(books_tokenized):
    os.makedirs(books_tokenized)

with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()

for i, filename in enumerate(os.listdir(books_utf_8)):

    if i < len(os.listdir(books_utf_8)):
        print("正在处理第", i + 1, "个文件{0}".format(filename))
        with open(os.path.join(books_utf_8, filename), "r+") as f:
        # with open(os.path.join(books_utf_8, filename), "r+", encoding="utf-8") as f:
            dst = []#[Start]
            w = f.read(1)#[Enter]
            while w:
                # print(w)
                if w == '\n' or w == '\r' or w == '\t' or ord(w) == 305:
                    dst.append("1")
                elif w == ' ':
                    dst.append("3")
                else:
                    try:
                        dst.append(str(tokens.index(w)))
                    except Exception:
                        dst.append("2")
                w = f.read(1)
        with open(os.path.join(books_tokenized, "{}".format(filename)), "w+", encoding="utf-8") as df:
            df.write(" ".join(dst))
print("完成所有文件的编码!")