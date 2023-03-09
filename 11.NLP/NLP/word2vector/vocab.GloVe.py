import torch
import torchtext
print(torch.__version__)
print(torchtext.__version__)
gv = torchtext.vocab.GloVe(name="6B",dim=50)#一个词用50个长度的向量来表示
print(len(gv.vectors))
print(gv.vectors.shape)#有40万个词向量，每个词向量长度为50
#
print(gv.stoi['tokyo'])#取出Tokyo的词索引
print(gv.vectors[1363])#再根据词索引取出词向量
print(len(gv.vectors[1363]))#一个词向量长度为50
# #这些词向量不是随机给的，是训练出来的，
# # 跟训练centerloss一样，只不过加载的这个词向量已经被训练好了
print(gv.itos[1363])#根据索引取出词


#取出一个词的向量
def get_wv(word):
    return gv.vectors[gv.stoi[word]]
print(get_wv("tokyo"))

#取出这个词向量，拿这个向量去遍历所有的向量，求距离，拿出10个最近的词
def sim_10(word,n = 10):
    '''欧式距离：torch.sqrt(torch.sum((hg1-hg2)**2)),torch.dist(hg1, hg2, p=2)'''
    #gv.itos[i]:取出索引为第i个的词,torch.dist(word,w): 求欧氏距离
    #拿到所有词与word的距离的元组组成的列表
    aLL_dists = [(gv.itos[i],torch.dist(word,w)) for i,w in enumerate(gv.vectors)]
    # print(aLL_dists[:10])
    # aLL_dists得到的是包含了两个元素的元组,第一个元素是词名称，第二个是这个词和指定词的距离
    #对得到的元组列表按距离从小到大排序，再取前10个查看，距离越小说明两个词向量越接近
    return sorted(aLL_dists,key=lambda t: t[1])[:n]
print(sim_10(gv.vectors[1363]))

def answer(w1,w2,w3):
    #"让中国词向量和北京的词向量距离等于 日本? 的词向量和东京的词向量距离"
    #相当于：china-beijing = japan-x，给网络输入三个值，求第四个值。
    print("{0}：{1}=={2}：{3}".format(w1,w2,w3,"x"))
    #w1-w2 = w3-w4
    #w4 = w3-w1+w2
    w4 = get_wv(w3)-get_wv(w1)+get_wv(w2)
    print(sim_10(w4))
    return sim_10(w4)[0][0]#拿出第一组的第一个值，也就是距离最小的词
print("x="+answer("china","beijing","japan"))
