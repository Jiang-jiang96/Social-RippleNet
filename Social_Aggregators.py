import torch
import torch.nn as nn
from Attention import Attention


class Social_Aggregator(nn.Module):

    def __init__(self, u_to_e, social_lists, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()
        self.u_to_e = u_to_e
        self.social_lists = social_lists
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.device = cuda

    def forward(self, nodes):

        neighs = []
        for node in nodes:
            neighs.append(self.social_lists[int(node)])

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            i_social = neighs[i]
            num_neighs = len(i_social)

            e_u_social = self.u_to_e.weight[list(i_social)]
            e_u = self.u_to_e.weight[nodes[i]]

            att_w = self.att(e_u_social, e_u, num_neighs)
            att_history = torch.mm(e_u_social.t(), att_w).t()
            embed_matrix[i] = att_history
        social_feature = embed_matrix

        return social_feature
