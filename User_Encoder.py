import torch
import torch.nn as nn
import torch.nn.functional as F


class User_Encoder(nn.Module):

    def __init__(self, args, uv_aggregator, social_aggregator, user_self, embed_dim, cuda="cpu"):
        super(User_Encoder, self).__init__()
        self._parse_args(args)
        self.uv_aggregator = uv_aggregator
        self.social_aggregator = social_aggregator
        self.user_self = user_self
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.device = cuda

    def _parse_args(self, args):
        self.interact_weight = args.interact_weight
        self.neigh_weight = args.neigh_weight
        self.self_weight = args.self_weight


    def forward(self, nodes):

        self_feature = self.user_self.forward(nodes)
        interact_feature = self.uv_aggregator.forward(nodes)
        neigh_feature = self.social_aggregator.forward(nodes)

        self_feature = self.self_weight * self_feature
        interact_feature = self.interact_weight * interact_feature
        neigh_feature = self.neigh_weight * neigh_feature

        combined = torch.cat([self_feature, interact_feature, neigh_feature], dim=1)
        encoder_u = F.relu(self.linear(combined))

        return encoder_u


