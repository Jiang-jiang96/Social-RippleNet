import torch
import torch.nn as nn
import torch.nn.functional as F


class RippleNet(nn.Module):
    def __init__(self, args, v2e, n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old2new):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation)
        self.v2e = v2e

        self.ripple_set = ripple_set
        self.history_not_kg_v = history_not_kg_v
        self.item_index_old2new = item_index_old2new

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.embed_dim
        self.n_hop = args.n_hop
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def forward(self, nodes):
        history_kg_feature = []
        for node in nodes:
            if node in self.history_not_kg_v:
                item_self = self.v2e.weight[int(node)]
                history_kg_feature.append(item_self.detach().numpy().tolist())
            else:
                v_index = self.item_index_old2new[int(node)]
                kg_data = self.ripple_set[v_index]
                memories_h, memories_r, memories_t = [], [], []
                for i in range(self.n_hop):
                    memories_h.append(torch.LongTensor([kg_data[i][0]]))
                    memories_r.append(torch.LongTensor([kg_data[i][1]]))
                    memories_t.append(torch.LongTensor([kg_data[i][2]]))

                # [dim]
                item_embeddings = self.entity_emb(node)
                h_emb_list = []
                r_emb_list = []
                t_emb_list = []
                for i in range(self.n_hop):
                    # [n_memory, dim]
                    h_emb_list.append(self.entity_emb(memories_h[i]))
                    # [n_memory, dim, dim]
                    r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
                    # [n_memory, dim]
                    t_emb_list.append(self.entity_emb(memories_t[i]))

                kg_feature = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)
                history_kg_feature.append(kg_feature.detach().numpy().tolist())

        return history_kg_feature

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        global kg_feature
        o_list = []
        for hop in range(self.n_hop):

            # [n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [n_memory, dim]   Rh = ri * pi
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [dim,1]
            v = torch.unsqueeze(item_embeddings, dim=-1)

            # [n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [n_memory]
            probs_normalized = F.softmax(probs, dim=0)

            # [n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=1)

            # [dim] = [n_memory, dim] * [n_memory, 1]
            o = torch.squeeze((t_emb_list[hop] * probs_expanded).sum(dim=1))

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)        # o_list = o1+o2+...

            kg_feature = o_list[-1]
            if self.using_all_hops:
                for i in range(self.n_hop - 1):
                    kg_feature += o_list[i]  # y=o1+o2+...

        return kg_feature

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":      # default = 'plus_transform'
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings


