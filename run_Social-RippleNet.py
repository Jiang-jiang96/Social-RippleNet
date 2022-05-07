import torch
import torch.nn as nn
import numpy as np
from UV_Aggregators import UV_Aggregator
from User_Embedding import User_Embedding
from Social_Aggregators import Social_Aggregator
from User_Encoder import User_Encoder
from Item_Encoder import Item_Encoder
from RippleNet import RippleNet
from data_loader import load_data
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import os


class Social_RippleNet(nn.Module):

    def __init__(self, encoder_u, encoder_v):
        super(Social_RippleNet, self).__init__()
        self.encoder_u = encoder_u
        self.encoder_v = encoder_v
        self.embed_dim = encoder_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.encoder_u(nodes_u)
        embeds_v = self.encoder_v(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():

    np.random.seed(555)
    parser = argparse.ArgumentParser(description='Social_RippleNet model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N', help='test batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use gpu')  # default = True

    parser.add_argument('--neigh_weight', type=float, default=0.3, help='weight of the KGE term/ social term')
    parser.add_argument('--interact_weight', type=float, default=0.3, help='weight of the interaction term')
    parser.add_argument('--self_weight', type=float, default=0.4, help='weight of the self term')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    data_file = open('./data/350000output.txt', 'r')
    a = data_file.read()
    data = eval(a)
    history_u_lists, history_v_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_lists, user_gender_list, user_age_list = data
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()    # max(history_u_lists)/history_u_lists.__len__()
    num_items = max(history_v_lists)    # max(history_v_lists) / history_v_lists.__len__()
    num_genders = 2
    num_ages = 7

    u_to_e = nn.Embedding(num_users+1, embed_dim).to(device)
    v_to_e = nn.Embedding(num_items+1, embed_dim).to(device)
    g_to_e = nn.Embedding(num_genders, embed_dim).to(device)
    a_to_e = nn.Embedding(num_ages, embed_dim).to(device)

    # get KG
    n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old_to_new = load_data(args)

    # user feature
    u_interact_feature = UV_Aggregator(v_to_e, u_to_e, history_u_lists, embed_dim, uv=True)
    u_social_feature = Social_Aggregator(u_to_e, social_lists, embed_dim)
    u_self_feature = User_Embedding(u_to_e, g_to_e, a_to_e, user_gender_list, user_age_list, embed_dim)
    # combine
    encoder_u = User_Encoder(args, u_interact_feature, u_social_feature, u_self_feature, embed_dim)

    # item feature:
    v_interact_feature = UV_Aggregator(v_to_e, u_to_e, history_v_lists, embed_dim, uv=False)
    # KG feature
    v_kg_feature = RippleNet(args, v_to_e, n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old_to_new)
    # combine
    encoder_v = Item_Encoder(args, v_to_e, v_interact_feature, v_kg_feature,  embed_dim)

    # model
    social_ripple_net = Social_RippleNet(encoder_u, encoder_v).to(device)
    optimizer = torch.optim.RMSprop(social_ripple_net.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(social_ripple_net, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(social_ripple_net, device, test_loader)

        # early stopping
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            print('early stopping')
            break


if __name__ == "__main__":
    main()
