import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.requires_grad = False

        # (L, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, T, N, D)
        # pe shape: (1, T, D) -> (1, T, 1, D) -> (B, T, N, D)
        pe = self.pe[:, :x.size(1)].unsqueeze(2).repeat(x.size(0), 1, x.size(2), 1)
        return pe.detach()


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=4, mask=False):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.mask = mask

        assert self.model_dim % self.num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = model_dim // num_heads

        # Q/K/V
        self.fc_Q = nn.Linear(model_dim, model_dim)
        self.fc_K = nn.Linear(model_dim, model_dim)
        self.fc_V = nn.Linear(model_dim, model_dim)
        # out
        self.out_proj = nn.Linear(model_dim, model_dim)  # FC

    def forward(self, query, key, value):
        # 1. query shape: (B, T, N, D) -> (B * num_heads, T, N, D / num_heads)
        batch_size = query.shape[0]
        tgt_len = query.shape[-2]
        src_len = key.shape[-2]

        query = self.fc_Q(query)
        key = self.fc_Q(key)
        value = self.fc_Q(value)

        # execute 1
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), 0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), 0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), 0)

        # A = Q * K^T
        key = key.transpose(-2, -1)
        att_score = query @ key / self.head_dim ** 0.5
        if self.mask:
            mask = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).tril()
            att_score.masked_fill_(~mask, -torch.inf)
        att_score = torch.softmax(att_score, -1)

        # Z = A * V: (B, T, N, N) * (B, T, N, D)
        out = att_score @ value
        out = torch.cat(torch.split(out, batch_size, 0), -1)
        out = self.out_proj(out)  # Y = FC(Z)
        return out.transpose(1, 2)


class Gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(Gcn, self).__init__()
        self.dropout = dropout
        self.order = order
        self.c_in = (support_len * order + 1) * c_in
        self.mlp = nn.Conv2d(self.c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x, support):
        # (B, T, N, D) -> (B, D, N, T)
        x = x.transpose(1, 3)
        out = [x]  # (B, D, N, T)

        # gcn operation
        for a in support:
            # A * X: (B, D, N, T)) * (N * N) -> (B, D, N, T)
            x1 = torch.einsum('ncvl, vw->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl, vw->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        h = torch.cat(out, 1)  # (B, D * len(out), N, T), 经过了gcn, D将会升维
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)

        return h.permute(0, 2, 3, 1)  # B, D, N, T -> B, T, N, D


class GTU(nn.Module):
    def __init__(self, model_dim, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.model_dim = model_dim
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv2out = nn.Conv2d(model_dim, 2 * model_dim, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        # x shape: (B, D, N, T)
        x_causal = self.conv2out(x)
        x_p = self.tanh(x_causal[:, :self.model_dim, :, :])
        x_q = self.sigmoid(x_causal[:, -self.model_dim:, :, :])
        x_gtu = torch.mul(x_p, x_q)

        return x_gtu


class TemporalConv(nn.Module):
    def __init__(self, model_dim, time_strides, horizon):
        super(TemporalConv, self).__init__()
        self.horizon = horizon
        self.model_dim = model_dim
        self.relu = nn.ReLU(inplace=True)

        self.gtu_2 = GTU(model_dim, time_strides, 2)
        self.gtu_3 = GTU(model_dim, time_strides, 3)
        self.gtu_5 = GTU(model_dim, time_strides, 5)
        self.gtu_7 = GTU(model_dim, time_strides, 7)

        self.fc = nn.Sequential(
            nn.Linear(4 * horizon - 13, horizon),  # 2 + 3 + 5 + 7 - 4 = 13
            nn.Dropout(0.05)
        )

    def forward(self, x):
        # x shape: (B, N, T, D) -> (B, D, N, T)
        feat_dim = x.shape[-1]
        x = x.permute(0, 3, 1, 2)

        x_gtu = [self.gtu_2(x), self.gtu_3(x), self.gtu_5(x), self.gtu_7(x)]
        time_conv = torch.cat(x_gtu, -1)  # (B, D, N , T), T = 4 * horizon - 13
        time_conv = self.fc(time_conv)  # (B, D, N , T)

        if feat_dim == 1:
            time_conv_out = self.relu(time_conv)
        else:
            time_conv_out = self.relu(x + time_conv)

        return time_conv_out.permute(0, 2, 3, 1)


class MemoryAugmented(nn.Module):
    def __init__(self, num_nodes, mem_num=100, mem_dim=32):
        super(MemoryAugmented, self).__init__()
        self.num_nodes = num_nodes
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.lamda = 0.7
        self.memory = self.construct_memory()

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        # Memory is Meta-Node Bank
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)

        for param in memory_dict.values():
            nn.init.xavier_uniform_(param)

        return memory_dict

    def query_memory(self, h):
        # h shape: (B, T, N, D)
        query = h

        # first Equation (7)
        att_score1 = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), -1)  # shape : (B, T, N, mem_num)
        value1 = torch.matmul(att_score1, self.memory['Memory'])  # value1 shape is : (B, T, N, D)

        # second Equation (7)
        att_score2 = torch.softmax(torch.matmul(value1, self.memory['Memory'].t()), -1)  # shape : (B, T, N, mem_num)
        value2 = torch.matmul(att_score2, self.memory['Memory'])  # value2 shape is : (B, T, N, D)

        # contrast loss
        _, ind = torch.topk(att_score2, k=2, dim=-1)
        pos = self.memory['Memory'][ind[..., 0]]
        neg = self.memory['Memory'][ind[..., 1]]

        return self.lamda * value1 + (1 - self.lamda) * value2, query, pos, neg

    def forward(self, x):
        # x shape: B T N D
        return self.query_memory(x)


class Model(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, minute_size,
                 num_layers=1, cheb_k=3, mem_num=20, mem_dim=64):
        super(Model, self).__init__()
        # required params setting
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_units = rnn_units
        self.minute_size = minute_size  # 288
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.embed_dim = 12
        self.weekday_size = 7
        self.adaptive_embedding_dim = 28
        self.model_dim = self.embed_dim * 3 + self.adaptive_embedding_dim  # 12 * 3 + 28 = 64

        # 4个embedding
        self.position_encoder = PositionalEncoding(self.embed_dim)
        self.daytime_embedding = nn.Embedding(self.minute_size, self.embed_dim)
        self.weekday_embedding = nn.Embedding(self.weekday_size, self.embed_dim)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.horizon, self.num_nodes, self.adaptive_embedding_dim))
        )

        # memory augmented
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory_aug = MemoryAugmented(self.num_nodes, self.mem_num, self.mem_dim)  # Meta-Node Bank

        # spatio-temporal layer -- 5 layers is the best
        self.fc = nn.Linear(self.input_dim, self.embed_dim)
        self.attentions = nn.ModuleList()
        self.gcnconvs = nn.ModuleList()
        self.tconvs = nn.ModuleList()
        self.ln = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attentions.append(AttentionLayer(self.model_dim, 4))
            self.gcnconvs.append(Gcn(self.model_dim, self.model_dim, 0.3, 2, self.cheb_k))
            self.tconvs.append(TemporalConv(self.model_dim, 1, self.horizon))
            self.ln.append(LayerNorm(self.model_dim))

        self.pred_conv = nn.Linear(self.model_dim * 2, self.model_dim)
        self.temporal_proj = nn.Linear(self.horizon, self.horizon)
        self.out_proj = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, x, x_conv):
        # x shape : (B, T, N, D), D = 1, only the speed dimension
        # x_conv shape: (B, T, N, D), D = 2, the daytime and weekday dimension

        # 0. 构建自适应邻接矩阵
        node_embeddings1 = torch.matmul(self.memory_aug.memory['We1'], self.memory_aug.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory_aug.memory['We2'], self.memory_aug.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        # 1. Data embedding layer
        x = self.fc(x)  # B, T, N, 1) -> B, T, N, 12), the speed embedding to 12 dimension
        # 1.1 Position encoding
        x = x + self.position_encoder(x)
        # 1.2 time of day and day of week embedding
        x_day = self.daytime_embedding((x_conv[..., 0] * self.minute_size).float().round().long())
        x_week = self.weekday_embedding((x_conv[..., 1].long()))
        # 1.3 adaptive embedding, expand dimension
        x_adaptive = self.adaptive_embedding.expand(size=(x.shape[0], *self.adaptive_embedding.shape))

        x = torch.cat((x, x_day, x_week, x_adaptive), dim=-1)  # shape is (B, T, N, D), D = 12 + 12 + 12 + 28 = 64
        x = x.transpose(1, 2)  # B, N, T, D

        # 2. spatio-temporal layer operation
        for i in range(self.num_layers):
            residual = x
            s = self.attentions[i](x, x, x)  # Q K V is all x
            x = self.gcnconvs[i](s, supports)
            x = self.tconvs[i](x)
            x = x + residual
            x = self.ln[i](x)
        x = x.transpose(1, 2)  # B, T, N, D

        # 3. memory augment from Mate-Node Bank
        x_aug, query, pos, neg = self.memory_aug(x)

        # 4. output prediction
        out = torch.cat((x, x_aug), dim=-1)  # 升维
        out = F.relu(self.pred_conv(out))  # 降维
        out = out.transpose(1, 3)  # (B, D, N, T)
        out = self.temporal_proj(out)
        out = self.out_proj(out.transpose(1, 3))  # (B, T, N, D) -> D = 1

        return out, query, pos, neg

