import collections
import os
import numpy as np



def load_data(args):
    n_entity, n_relation, kg = load_kg()
    history_kg_v, history_not_kg_v, item_index_old2new = v_list()
    ripple_set = get_ripple_set(args, kg, history_kg_v)
    return n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old2new


def load_kg():
    print('reading KG file ...')

    # reading kg file
    kg_file = './data/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))  # kg={head:[tail,relation]}
    return kg


def v_list():
    item_index_old2new = dict()
    file = './data/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        item_index_old2new[int(item_index)] = i
        i += 1

    data_file = open('./data/350000output.txt', 'r')
    a = data_file.read()
    data = eval(a)
    history_v = data[1]
    history_v_lists = list(history_v.keys())
    history_not_kg_v = []
    history_kg_v = []

    for v in history_v_lists:
        if v not in item_index_old2new:
            history_not_kg_v.append(v)
        else:
            v_index = item_index_old2new[v]
            history_kg_v.append(v_index)

    return history_kg_v, history_not_kg_v, item_index_old2new


def get_ripple_set(args, kg, history_kg_v):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)  # ripple_set = {movie:[memories_h, memories_r, memories_t]}

    for node in history_kg_v:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = [node]
            else:
                tails_of_last_hop = ripple_set[node][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:  # kg = {head:[tail,relation]}
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            if len(memories_h) == 0:
                ripple_set[node].append(ripple_set[node][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[node].append((memories_h, memories_r, memories_t))

    return ripple_set
