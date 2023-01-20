import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda')

###########################################

# TGAT模型

###########################################

class GraphAttentionLayer(nn.Module):


    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414).cuda()
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414).cuda()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        batch_size_adj = h.size(0)
        zero_vec = -9e15 * torch.ones_like(e)

        adj = torch.tensor(np.expand_dims(adj.cpu(), 0).repeat(batch_size_adj, axis=0))

        attention = torch.where(adj > 0, e.cpu(), zero_vec.cpu())
        attention = F.softmax(attention.cuda(), dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)


        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = Wh2.transpose(1, 2)

        e = Wh1 + Wh2
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.dropout = 0.2
        self.alpha = 0.2
        self.concat = True

        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()


        self.graphAttentionLayer = GraphAttentionLayer((num_gru_units + 1),
                                                       (num_gru_units + 1), self.dropout,
                                                       self.alpha, self.concat)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, adj, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        a_times_concat = self.graphAttentionLayer(concatenation, adj)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim
        )

    def forward(self, adj, inputs, hidden_state):
        concatenation = torch.sigmoid(self.graph_conv1(adj, inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(adj, inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN_model(nn.Module):
    def __init__(self, hidden_dim: int):
        super(TGCN_model, self).__init__()
        self._input_dim = 14
        self._hidden_dim = hidden_dim
        self.tgcn_cell = TGCNCell(self._input_dim, self._hidden_dim)

    def forward(self, adj, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(adj, inputs[:, i, :], hidden_state)

            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


def calculate_laplacian_with_self_loop(matrix):
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class TGAT(nn.Module):
    def __init__(
            self,
            hidden_dim,
            dropout=0.2,
            num_nodes=14
    ):
        super(TGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.model = TGCN_model(hidden_dim)
        self.dropout = dropout

        # section3
        self.dense_weight = nn.Linear(hidden_dim, 1)
        self.regressor = (
            nn.Linear(self.hidden_dim, 1))




    def forward(self, adj, x):
        batch_size, _, num_nodes = x.size()
        hidden = self.model(adj, x)
        weights = F.softmax(self.dense_weight(hidden), dim=1)
        outputs = torch.sum(hidden * weights, dim=1)

        predictions = self.regressor(outputs)


        return predictions, hidden, outputs




class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0.2, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, batch_size):
        z = F.dropout(z, self.dropout, training=self.training)
        z1 = F.normalize(z, p=2, dim=2).cuda()
        z2 = torch.bmm(z1, z1.swapaxes(1, 2))
        adj = self.act(z2)

        return adj

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)