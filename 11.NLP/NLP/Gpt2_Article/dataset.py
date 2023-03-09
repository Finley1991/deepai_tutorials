import Gpt2_Article.config as cfg
import torch
import os
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dir):

        self.dataset = []

        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), "r+") as f:
                #读出语料库的索引表token里的位置编码，放在列表里面：611
                ws = [int(x) for x in f.readline().split()]
                print(len(ws))#2762
                ws_len = len(ws)
                start = 0
                #从语料库中当拿到句子长度大于等于1200个时候，才算作训练样本
                while ws_len - start >= cfg.pos_num:
                    #将语料库中的文字按照1200的长度为一条样本切分成样本集
                    #ws[0:1200],ws[40:1200-40],......
                    self.dataset.append(ws[start:start + cfg.pos_num])
                    # print(ws[start:start + cfg.pos_num])
                    # 步长表示每条样本之间在语料库中的相隔的字数，
                    # 步长越小学习的结果越多样化，样本也越多；步长越大，样本集越少
                    #此处样本集步长为40，表示每隔40 个字取一次样本
                    # start += 40
                    #步长为1，样本最多
                    start += 1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index])
        #[0:1119],[1:1200]
        return data[0:-1], data[1:]


if __name__ == '__main__':
    myDataset = MyDataset("./data/books_tokenized")
    print(len(myDataset))
    print(myDataset[0])
    print(myDataset[1])
