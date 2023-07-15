import torch
import torchtext

gv = torchtext.vocab.GloVe(name="6B",dim=50)#一个词用50个长度的向量来表示

def get_wv(word):
    return gv.vectors[gv.stoi[word]]

def sim_10(word,n = 10):
    aLL_dists = [(gv.itos[i],torch.dist(word,w)) for i,w in enumerate(gv.vectors)]
    return sorted(aLL_dists,key=lambda t: t[1])[:n]

def answer(w1,w2,w3):
    print("{0}：{1}=={2}：{3}".format(w1,w2,w3,"x"))
    w4 = get_wv(w3)-get_wv(w1)+get_wv(w2)
    print(sim_10(w4))
    return sim_10(w4)[0][0]#拿出10组中的第一组的第一个值，也就是距离最小的词
print("x="+answer("china","beijing","japan"))
