import os


DIR_PATHS = [r"./books_convert"]
VOCAB_FILE = "../data/vocab.txt"


def main():
    words = set()
    for DIR_PATH in DIR_PATHS:
        for i, filename in enumerate(os.listdir(DIR_PATH)):
            f_path = os.path.join(DIR_PATH, filename)
            with open(f_path, "r+") as f:
            # with open(f_path, "r+", encoding="utf-8") as f:
                w = f.read(1)#取文章中长度为1 的第一个字
                while w:

                    if w == '\n' or w == '\r' or w == '\t' or w == ' ':
                        pass
                    else:
                        words.add(w)
                    #改变循环条件，读取下一个字
                    w = f.read(1)

    with open(VOCAB_FILE, "w+", encoding="utf-8") as f:
        f.write("[Unk] [Ent] [Spa] ")
        f.write(" ".join(words))
        f.flush()
        print("字去重已完成！")


main()
