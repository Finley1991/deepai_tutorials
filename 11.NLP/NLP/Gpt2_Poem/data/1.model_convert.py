import codecs
import os

books_original_path = r"./books_original"
books_convert_path = r"./books_convert"


# noinspection PyArgumentEqualDefault
def ReadFile(filePath, encoding=""):
    with codecs.open(filePath, 'r', encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding=""):
    with codecs.open(filePath, "w", encoding) as f:
        f.write(u)



def UTF8_2_GBK(src, dst):
    content = ReadFile(src, encoding="utf-8")
    WriteFile(dst, content, encoding="gb18030")


def GBK_2_UTF8(src, dst):
    content = ReadFile(src, encoding="gb18030")
    WriteFile(dst, content, encoding="utf-8")


def main():
    model = input("请选择要转换的模式结果, 输入U可以转换为UTF-8,输入G可以转换为GBK：\n")
    if model == "U" or model =="G":
        for i, filename in enumerate(os.listdir(books_original_path)):
            original_path = os.path.join(books_original_path, filename)
            convert_path = os.path.join(books_convert_path, filename)
            if model == 'G':
                """++++++++UTF8-2-gbk+++++++++"""
                UTF8_2_GBK(original_path, convert_path)
            elif model == "U":
                """++++++++GBK-2-utf8+++++++++"""
                GBK_2_UTF8(original_path, convert_path)
            print("转换完成")
    else:
        print("您的输入是{0}, 输入不正确，请重新输入大写G或大写U".format(model))


if __name__ == '__main__':
    main()
