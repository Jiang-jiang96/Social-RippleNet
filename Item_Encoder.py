import torch
import torch.nn as nn
import torch.nn.functional as F


class Item_Encoder(nn.Module):

    def __init__(self, args, v_to_e, uv_aggregator, graph_aggregator, embed_dim, cuda="cpu"):
        super(Item_Encoder, self).__init__()
        self._parse_args(args)
        self.v_to_e = v_to_e
        self.uv_aggregator = uv_aggregator
        self.graph_aggregator = graph_aggregator
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.device = cuda

    def _parse_args(self, args):
        self.interact_weight = args.interact_weight
        self.neigh_weight = args.neigh_weight
        self.self_weight = args.self_weight

    def forward(self, nodes):

        self_feature = self.v_to_e.weight[nodes]
        interact_feature = self.uv_aggregator.forward(nodes)
        graph_feature_list = self.graph_aggregator.forward(nodes)
        graph_feature = torch.LongTensor(graph_feature_list)

        self_feature = self.self_weight * self_feature
        interact_feature = self.interact_weight * interact_feature
        neigh_feature = self.neigh_weight * graph_feature

        combined = torch.cat([self_feature, interact_feature, neigh_feature], dim=1)
        encoder_v = F.relu(self.linear(combined))

        return encoder_v


