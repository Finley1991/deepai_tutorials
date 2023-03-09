import torch.nn as nn
import Gpt2_Poem.config as cfg
import torch

class Attention(nn.Module):

    def __init__(self, isMask=True):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5#60
        self.isMask = isMask
        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)   # 扩充为3v（q，k，v）:720,720*3
        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        #学习注意力
        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)    # 线性层

        if self.isMask:
            #生成下三角的一矩阵:返回一个矩阵主对角线以下的下三角矩阵，其它元素全部为0。
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))


    def forward(self, x):    # x：NSV

        # print(x.shape) #N,51,720
        x = self.c_attn(x)#NS(3V):N,51,720*3

        #【N*51,12，720*3/12】，将单头变成多头
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)   # N，51，12，180（NSHV）
        #把头放在前面
        x = x.transpose(-2, -3)  # NHSV:N.12,51,180
        #只把词向量V分成三份，其他的复制
        q, k, v = x.chunk(3, dim=-1)#[10, 12, 51, 60]
        # print(q.shape,k.shape,v.shape)

        #softmax之前的注意力系数，张量乘法需要转置NHSV@NHVS=NHSS，
        w = (q @ k.transpose(-1, -2)) / self.dk #NHSS
        if self.isMask:
            #NMSS
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]#【S,S】
            # print(mask.shape)
            # print(mask)
            # 掩码矩阵，生成左下角为特征，右上角为负无穷
            w = w * mask - (1 - mask) * 1e5
            # print(w*mask)
            # print((1-mask)*1e5)
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)

        # NHSS*NHSV=NHSV
        a = w @ v   # NHSV（N，H，S，60）
        # print(a.shape)#[10, 12, 51, 60]
        #要转成NSV的格式还回去
        a = a.transpose(-2, -3)   # NHSV-->NSHV
        # 把最后两个维度转成720维：12*60（H*V）
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)   # NSV
        # print(a.shape)#[10, 51, 720]
        h = self.c_proj(a)
        h = self.resi_drop(h)

        return h


class Block(nn.Module):

    def __init__(self, isMask):
        super().__init__()
        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)
        #调用注意力
        self.attention = Attention(isMask)
        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim,  2 * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * cfg.embed_dim, cfg.embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        #做残差
        a = a + x
        a = self.layer_normal_2(a)
        h = self.proj(a)
        h = self.dropout(h)
        y = h + a
        return y


class GPT2(nn.Module):

    def __init__(self,isMask):
        super().__init__()
        #定义词向量的维度:[305，720]
        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        #定义位置编码：[52,720]
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)

        self.blocks = []
        # 层数：12
        for _ in range(cfg.block_num):
            self.blocks.append(Block(isMask))
        self.drop = nn.Dropout(0.1)
        #调用block块，做一次前向
        self.sequential = nn.Sequential(*self.blocks)
        #输出词维度【720，305】
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, x, p):
        #获取向量
        e = self.vocab_embed(x)
        #获取位置
        p = self.pos_embed(p)
        # 把向量和位置加起来，一起做前向
        h = self.drop(e + p)
        h = self.sequential(h)
        return self.output_layer(h)

if __name__ == '__main__':
    data=torch.randn(10,51,720)
    x=torch.randint(0,100,(10,51))
    p=torch.arange(0,51).repeat(10).reshape(10,-1)
    print(x.shape)
    print(p.shape)
    att=Attention(True)
    att(data)
    gpt2=GPT2(True)
    out=gpt2(x,p)
    print(out.shape)