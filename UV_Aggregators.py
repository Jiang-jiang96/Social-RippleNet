import torch
import torch.nn as nn
from Attention import Attention


class UV_Aggregator(nn.Module):

    def __init__(self, v_to_e, u_to_e, history_uv_lists, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v_to_e = v_to_e
        self.u_to_e = u_to_e
        self.history_uv_lists = history_uv_lists
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.device = cuda

    def forward(self, nodes):

        history_uv = []
        for node in nodes:
            history_uv.append(self.history_uv_lists[int(node)])

        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_interact = len(history)

            if self.uv == True:
                e_uv_interact = self.v_to_e.weight[history]
                e_uv = self.u_to_e.weight[nodes[i]]
            else:
                e_uv_interact = self.u_to_e.weight[history]
                e_uv = self.v_to_e.weight[nodes[i]]

            att_w = self.att(e_uv_interact, e_uv, num_interact)
            att_history = torch.mm(e_uv_interact.t(), att_w)
            att_history = att_history.t()
            embed_matrix[i] = att_history
        neigh_feature = embed_matrix
        return neigh_feature
