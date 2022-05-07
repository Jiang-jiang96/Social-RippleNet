import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()

        self.embed_dim = embedding_dims
        self.liner1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.liner2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.liner3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node_uv, uv, neighs_num):

        uv_rep = uv.repeat(neighs_num, 1)
        x = torch.cat((node_uv, uv_rep), 1)
        x = F.relu(self.liner1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.liner2(x))
        x = F.dropout(x, training=self.training)
        x = self.liner3(x)
        attention = F.softmax(x, dim=0)

        return attention
