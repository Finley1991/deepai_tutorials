import torch.nn as nn
import Gpt2_Article.config as cfg
import torch


class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        #根号下词维度，缩小注意力的系数，这里除了头数：（720/12）**0.5
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask
        #转成符合多头注意力的向量参数（扩充3倍），720*3
        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)   # 扩充为3v（q，k，v）
        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        #学习注意力
        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)    # 线性层

        if self.isMask:
            #生成下三角的一矩阵，随着模型设备传输，但是不进行优化学习，【24，24】
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, x):    # x：NSV
        #N,24,720*3
        x = self.c_attn(x)#NS(3V)
        #【NS,12，720*3/12】，将单头变成多头
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)   # N，S，12，60*3（NSHV）
        #把头放在前面，每个词的注意力是分在多个头上的
        x = x.transpose(-2, -3)  # NHSV
        #每个词向量V分成三份（还原成原来的参数：（60*3）/3），每份的形状还是NHSV
        q, k, v = x.chunk(3, dim=-1)

        #softmax之前的注意力系数，张量乘法需要转置NHSV@NHVS=NHSS，
        # 前面维度保持不变，后面两个维度满足矩阵乘法规则
        w = (q @ k.transpose(-1, -2)) / self.dk #NHSS
        if self.isMask:
            #NMSS，S是一句话里字的个数：24
            #取出之前存的mask矩阵，切片后获得24*24的下三角掩码矩阵
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]#【S,S】
            # w乘以掩码矩阵，得到上部分为0，下部分不变，1-mask为上三角矩阵，上面为1，下面为0
            # (1 - mask) * 1e5得到一个上面为1e5，下面为0的上三角矩阵
            w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)
        #用注意力系数乘以向量的值，得到注意力
        # NHSS*NHSV=NHSV
        a = w @ v   # NHSV（N，H，S，60）

        #要转成NSV的格式还回去
        a = a.transpose(-2, -3)   # NHSV-->NSHV
        # 把最后两个维度转成720维：12*60（H*V）
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)   # NSV

        h = self.c_proj(a)
        h = self.resi_drop(h)

        return h


class Block(nn.Module):

    def __init__(self, isMask=False):
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

    def __init__(self):
        super().__init__()
        #定义词向量的维度:[307，720]
        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        #定义位置编码：[24,720]
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)

        self.blocks = []
        # 层数：12
        for _ in range(cfg.block_num):
            self.blocks.append(Block(isMask=False))
        self.drop = nn.Dropout(0.1)
        #调用block块，做一次前向
        self.sequential = nn.Sequential(*self.blocks)
        #输出词维度【720，307】
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

# class Attention(nn.Module):
#     def __init__(self, isMask=True):
#         super(Attention, self).__init__()
#
#         # dk = 词的维度/词的头数
#         self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
#
#         # 把一个词分解成Q，K，V
#         self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
#
#         self.attn_drop = nn.Dropout(0.1)
#         self.resi_drop = nn.Dropout(0.1)
#
#         # 接一个线性曾提供参数，使得词向量可训练
#         self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
#
#         self.isMask = isMask
#         if self.isMask:
#             # 定义一个下三角掩码，写在这儿会当成权重保存，不用被训练，自动传入网络
#             self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))
#         # 如果掩码这样写，需要手动传入数据
#         # self.mask = (数据).cuda()
#
#     def forward(self, x):
#         # x形状(N,S,V)，N代表多少个句子，S代表多少个词，V代表每个词的维度
#         x = self.c_attn(x)
#
#         # (N,S,V)——>(N,S,H,V)(N,S, H, V/H*3)
#         x = x.reshape(*x.shape[:-1], cfg.head_num, -1)
#
#         # (N,S,H,V)(N,S,H,V/H*3)——>(N,H,S,V)(N,H,S,V/H*3)
#         x = x.transpose(-2, -3)
#
#         # (N,H,S,V)(N,H,S,V/H*3) ——>(N,H,S,V)(N,H,S,V/H))
#         q, k, v = x.chunk(3, dim=-1)
#
#         # (N,H,S,(V/H))@(N,H,(V/H),S)=(N,H,S,S)
#         w = (q @ k.transpose(-1, -2)) / self.dk
#
#         # 掩码形状（S,S）
#         if self.isMask:
#             mask = self.mask[0:w.size(-2), 0:w.size(-1)]
#             # 将w的上三角全部变为负无穷小，有利于做softmax归一化
#             w = w * mask - (1 - mask) * 1e8
#
#         # 归一化得到权重
#         w = torch.softmax(w, dim=-1)
#
#         # dropout
#         w = self.attn_drop(w)
#
#         # (N,H,S,S)@(N,H,S,(V/H))-->(N,H,S,V)(N,H,S,(V/H))
#         a = w @ v
#
#         """和合并形状"""
#         # (N,H,S,(V/H))-->(N,S,H,(V/H))
#         a = a.transpose(-2, -3)
#         # (N,S,H,(V/H))-->(N,S,V)
#         a = a.reshape(*a.shape[:-2], cfg.embed_dim)
#
#         # 全连接层提供参数
#         h = self.c_proj(a)
#
#         # dropout
#         h = self.resi_drop(h)
#
#         return h
#
#
# class Block(nn.Module):
#
#     def __init__(self):
#         super(Block, self).__init__()
#
#         # 数据传进来归一化
#         self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)
#
#         # 注意力
#         self.attention = Attention()
#
#         # 值控制到0~1
#         self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)
#
#         # 全连接层，扩大参数量
#         self.proj = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
#                                   nn.LeakyReLU(),
#
#                                   nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim)
#                                   )
#
#         # dropout
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         h = self.layer_normal_1(x)
#         a = self.attention.forward(h)
#
#         # 加一个残差
#         a = a + x
#
#         a = self.layer_normal_2(a)
#
#         h = self.proj(a)
#
#         h = self.dropout(h)
#
#         # 加一个残差
#         y = h + a
#
#         return y
#
#
# class GPT2(nn.Module):
#
#     def __init__(self):
#         super(GPT2, self).__init__()
#
#         # 定义一个字典
#         self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
#
#         # 定义一个位置编码
#         self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)
#
#         # 定义一个类型编码
#         self.type_embed = nn.Embedding(cfg., cfg.embed_dim)
#
#         # 叠6层block
#         self.blocks = []
#         for _ in range(cfg.block_num):
#             self.blocks.append(Block())
#
#         # dropout
#         self.drop = nn.Dropout(0.1)
#
#         # 将叠的block形成一个网络
#         self.sequential = nn.Sequential(*self.blocks)
#
#         # 全连接输出层
#         self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)
#
#     def forward(self, x, p, ):
#         # 对输入进行词向量编码
#         e = self.vocab_embed(x)
#
#         # 对输入进行位置编码
#         p = self.pos_embed(p)
#
#         # # 对输入进行类型编码
#         # t = self.type_embed(t)
#         # h = self.drop(e + p + t)
#         h = self.drop(e + p)
#
#         h = self.sequential(h)
#
#         return self.output_layer(h)
