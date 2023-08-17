import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear

class SGCN_Ori(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3, class_num=2, hidden_size=64, rois=90):
        super(SGCN_Ori, self).__init__()
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.prob_dim = H_0
        self.dim1 = self.rois * H_3 + self.rois * H_2  # H_1*2 + H_3*2#hp.N//4*H_3  #self.rois*H_3      self.rois*H_3 +  self.rois*H_2
        self.dim2 = 64  # 64#128  64
        self.dim3 = 16  # 16#32 16

        # layers
        self.conv1 = GCNConv(H_0, H_1)
        self.conv2 = GCNConv(H_1, H_2)
        self.conv3 = GCNConv(H_1, H_3)
        # self.fc1 = torch.nn.Linear(self.rois * H_3, class_num)#29696 51200 59392
        self.fc1 = torch.nn.Linear(self.dim1, self.dim2)  # self.rois*H_3
        # self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        # self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, class_num)
        # self.h1_weights = torch.nn.Linear(59392, hidden_size)
        # self.h2_weights = torch.nn.Linear(hidden_size, class_num)

        self.fc1 = torch.nn.Linear(self.dim1, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, class_num)

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))
        # weights_init(self)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob) #self.prob
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)
        # print(x_prob)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss

        return loss_prob

    def forward(self, data, isExplain=False):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h0.requires_grad = True
        self.input = h0

        if isExplain:
            h0_prob, edge_weight_prob, _, _ = self.cal_probability(h0, edge_index, edge_weight)
        else:
            h0_prob, edge_weight_prob = h0, edge_weight
        h1 = F.relu(self.conv1(h0_prob, edge_index, edge_weight_prob))
        # h2 = F.relu(self.conv2(h1, edge_index, edge_weight_prob))
        h2 = h1
        with torch.enable_grad():
            self.final_conv_acts = self.conv3(h2, edge_index, edge_weight_prob)
        self.final_conv_acts.register_hook(self.activations_hook)
        h3 = F.relu(self.final_conv_acts)

        fill_value = h2.min().item() - 1
        batch_x, _ = to_dense_batch(h2, data.batch, fill_value)
        B, N, D = batch_x.size()
        z1 = batch_x.view(B, -1)

        fill_value = h3.min().item() - 1
        batch_x, _ = to_dense_batch(h3, data.batch, fill_value)
        B, N, D = batch_x.size()
        z2 = batch_x.view(B, -1)
        #
        final_z = torch.cat((z1, z2), 1)

        x = final_z

        # x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.7, training=self.training)
        logits = F.log_softmax(self.fc3(x), dim=-1)  # F.logsigmoid(self.fc3(x)) #F.log_softmax(self.fc3(x), dim=-1)
        return logits

    def __repr__(self):
        return self.__class__.__name__


class SGCN_GAT(torch.nn.Module):

    def __init__(self, dataset, num_layers, hidden, *args, hidden_linear=64, rois=90, H_0=3, **kwargs):
        super(SGCN_GAT, self).__init__()
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.prob_dim = H_0
        self.conv1 = GATConv(dataset.num_features, hidden, edge_dim=1)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden, edge_dim=1))
        self.lin1 = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2 = Linear(hidden_linear, dataset.num_classes)

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob) #self.prob
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)
        # print(x_prob)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss

        return loss_prob

    def forward(self, data, isExplain=False):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x.requires_grad = True
        self.input = x

        if isExplain:
            x_prob, edge_weight_prob, _, _ = self.cal_probability(x, edge_index, edge_weight)
        else:
            x_prob, edge_weight_prob = x, edge_weight

        x = F.relu(self.conv1(x_prob, edge_index, edge_weight_prob))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight_prob))
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        x = torch.cat(xs, dim=1)

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()
        z2 = batch_x.view(B, -1)
        x = z2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class SGCN_GCN(torch.nn.Module):

    def __init__(self, dataset, num_layers, hidden, *args, hidden_linear=64, rois=90, H_0=3, num_features=3, num_classes=2, **kwargs):
        super(SGCN_GCN, self).__init__()
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.prob_dim = H_0
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob) #self.prob
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)
        # print(x_prob)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss

        return loss_prob

    def forward(self, data, isExplain=False):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x.requires_grad = True
        self.input = x

        if isExplain:
            x_prob, edge_weight_prob, _, _ = self.cal_probability(x, edge_index, edge_weight)
        else:
            x_prob, edge_weight_prob = x, edge_weight

        x = F.relu(self.conv1(x_prob, edge_index, edge_weight_prob))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight_prob))
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        x = torch.cat(xs, dim=1)

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()
        z2 = batch_x.view(B, -1)
        x = z2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__