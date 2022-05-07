import torch
import torch.nn as nn
import torch.nn.functional as F


class User_Embedding(nn.Module):

    def __init__(self, u_to_e, g_to_e, a_to_e, user_gender_list, user_age_list, embed_dim):
        super(User_Embedding, self).__init__()
        self.u_to_e = u_to_e
        self.g_to_e = g_to_e
        self.a_to_e = a_to_e
        self.user_gender_list = user_gender_list
        self.user_age_list = user_age_list
        self.embed_dim = embed_dim
        self.linear = nn.Linear(self.embed_dim * 3, self.embed_dim)

    def forward(self, nodes):

        node_gender = []
        node_age = []
        for node in nodes:
            node_gender.append(self.user_gender_list[int(node)])
            node_age.append(self.user_age_list[int(node)])

        e_g = self.g_to_e.weight[node_gender]
        e_a = self.a_to_e.weight[node_age]
        e_u = self.u_to_e.weight[nodes]

        x = torch.cat((e_g, e_a, e_u), 1)
        self_feats = F.relu(self.linear(x))

        return self_feats
