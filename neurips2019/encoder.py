import torch, math
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
  """Two-layer fully-connected ELU net with batch norm."""

  def __init__(self, n_in, n_hid, n_out, do_prob=0.):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(n_in, n_hid)
    self.fc2 = nn.Linear(n_hid, n_out)
    self.bn = nn.BatchNorm1d(n_out)
    self.dropout_prob = do_prob

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.1)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def batch_norm(self, inputs):
    if len(inputs.shape) == 3:
      x = inputs.view(inputs.size(0) * inputs.size(1), -1)
      x = self.bn(x)
      return x.view(inputs.size(0), inputs.size(1), -1)
    elif len(inputs.shape) == 2:
      return self.bn(inputs)
    else: raise NotImplementedError

  def forward(self, inputs):
    # Input shape: [num_sims, num_things, num_features]
    x = F.elu(self.fc1(inputs))
    x = F.dropout(x, self.dropout_prob, training=self.training)
    x = F.elu(self.fc2(x))
    return self.batch_norm(x)


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.1)

  def edge2node(self, x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    incoming = torch.matmul(rel_rec.t(), x)
    return incoming / incoming.size(1)

  def node2edge(self, x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([receivers, senders], dim=2)
    return edges

  def forward(self, inputs, rel_rec, rel_send):
    raise NotImplementedError


class FullMLPStructureEncoder(Encoder):
  def __init__(self, n_in, n_hid, n_out, n_nodes=5, do_prob=0.):
    super(FullMLPStructureEncoder, self).__init__()
    self.n_e = 8
    # self.n_hid = n_hid
    self.n_nodes = n_nodes
    self.n_edges = self.n_nodes * (self.n_nodes-1)
    self.feat = self.n_edges * self.n_e
    self.mlp1 = MLP(n_in, self.n_e, self.n_e, do_prob)
    self.mlp2 = MLP(self.feat, 2*self.feat, self.feat, do_prob)
    self.mlp3 = MLP(self.feat, 2*self.feat, self.feat, do_prob)
    self.mlp4 = MLP(self.feat, 2*self.feat, self.feat, do_prob)
    print("Using structure encoder, hidden size {}.".format(n_hid))
    self.fc_out = nn.Linear(self.n_e, n_out)
    self.init_weights()

  def forward(self, inputs, rel_rec, rel_send):
    x = inputs.view(inputs.size(0), inputs.size(1), -1)
    # New shape: [num_sims, num_edges, num_edge_modules]
    bs = x.shape[0]
    x = self.mlp1(x).reshape(bs, -1).contiguous()
    after1 = x
    x = self.mlp2(x)
    x = x + after1
    after2 = x
    x = self.mlp3(x)
    x = x + after2
    after3 = x
    x = self.mlp4(x)
    x = (x + after3).reshape(bs, self.n_edges, self.n_e)
    return F.softmax(self.fc_out(x), dim=2)  # [:,:,1]

class MLPStructureEncoder(Encoder):
  def __init__(self, n_in, n_hid, n_out, do_prob=0.):
    super(MLPStructureEncoder, self).__init__()
    n_hid = 32
    self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
    self.mlp15 = MLP(n_hid, n_hid, n_hid, do_prob)
    self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
    self.mlp3 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
    self.mlp4 = MLP(n_hid * 4, n_hid, n_hid, do_prob)
    print("Using structure encoder, hidden size {}.".format(n_hid))
    self.fc_out = nn.Linear(n_hid, n_out)
    self.init_weights()

  def forward(self, inputs, rel_rec, rel_send):
    x = inputs.view(inputs.size(0), inputs.size(1), -1)
    # New shape: [num_sims, num_edges, num_edge_modules]
    x = self.mlp1(x)  # 2-layer ELU net per node
    first_edges = x
    x = self.edge2node(x, rel_rec, rel_send)
    x = self.mlp15(x)
    x = self.node2edge(x, rel_rec, rel_send)
    x = torch.cat((self.mlp2(x) , first_edges), dim=2)
    x_skip = x
    x = self.edge2node(x, rel_rec, rel_send)
    x = self.mlp3(x)
    x = self.node2edge(x, rel_rec, rel_send)
    x = torch.cat((x, x_skip), dim=2)  # Skip connection
    x = self.mlp4(x)

    return F.softmax(self.fc_out(x), dim=2)  # [:,:,1]


class MLPStateEncoder(Encoder):
  def __init__(self, n_in, n_hid, n_out, do_prob=0.):
    super(MLPStateEncoder, self).__init__()
    self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
    self.bn1 = nn.BatchNorm1d(n_hid)
    self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
    self.bn2 = nn.BatchNorm1d(n_hid)
    self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
    self.bn3 = nn.BatchNorm1d(n_hid)
    self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
    self.bn4 = nn.BatchNorm1d(n_hid)
    print("Using state encoder, hidden size {}.".format(n_hid))
    self.n_hid = n_hid
    self.fc_out = nn.Linear(n_hid, n_out)
    self.init_weights()

  def forward(self, inputs, rel_rec, rel_send):
    # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
    x = inputs.view(inputs.size(0), inputs.size(1), -1)
    # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
    bs = x.shape[0]
    x = self.mlp1(x)  # 2-layer ELU net per node
    x = self.bn1(x.reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)
    x = self.node2edge(x, rel_rec, rel_send)
    x = self.mlp2(x)
    x = self.bn2(x.reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)
    x_skip = x
    x = self.edge2node(x, rel_rec, rel_send)
    x = self.mlp3(x)
    x = self.bn3(x.reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)
    x = self.node2edge(x, rel_rec, rel_send)
    x = torch.cat((x, x_skip), dim=2)  # Skip connection
    x = self.mlp4(x)
    x = self.bn4(x.reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)

    return F.softmax(self.fc_out(x), dim=2)#[:,:,1]

##############
#    CNN     #
##############

class CNN(nn.Module):
  def __init__(self, n_in, n_hid, n_out, do_prob=0.):
    super(CNN, self).__init__()
    self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                             dilation=1, return_indices=False,
                             ceil_mode=False)

    self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
    self.bn1 = nn.BatchNorm1d(n_hid)
    self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
    self.bn2 = nn.BatchNorm1d(n_hid)
    self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
    self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
    self.dropout_prob = do_prob

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.fill_(0.1)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  @staticmethod
  def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

  def forward(self, inputs):
    # Input shape: [num_sims * num_edges, num_dims, num_timesteps]
    x = F.relu(self.conv1(inputs))
    x = self.bn1(x)
    x = F.dropout(x, self.dropout_prob, training=self.training)
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.bn2(x)
    pred = self.conv_predict(x)
    attention = self.my_softmax(self.conv_attention(x), axis=2)

    edge_prob = (pred * attention).mean(dim=2)
    return edge_prob


class CNNStateEncoder(Encoder):
  def __init__(self, n_in, n_hid, n_out, do_prob=0.):
    super(CNNStateEncoder, self).__init__()
    self.dropout_prob = do_prob

    self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
    self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
    self.bn1 = nn.BatchNorm1d(n_hid)
    self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
    self.bn2 = nn.BatchNorm1d(n_hid)
    self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
    self.bn3 = nn.BatchNorm1d(n_hid)
    self.fc_out = nn.Linear(n_hid, n_out)
    self.n_hid = n_hid

    print("Using CNN state encoder, hidden size {}.".format(n_hid))

    self.init_weights()

  def node2edge_temporal(self, inputs, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.

    x = inputs.view(inputs.size(0), inputs.size(1), -1)

    receivers = torch.matmul(rel_rec, x)
    receivers = receivers.view(inputs.size(0) * receivers.size(1),
                               inputs.size(2), inputs.size(3))
    receivers = receivers.transpose(2, 1)

    senders = torch.matmul(rel_send, x)
    senders = senders.view(inputs.size(0) * senders.size(1),
                           inputs.size(2),
                           inputs.size(3))
    senders = senders.transpose(2, 1)

    # receivers and senders have shape:
    # [num_sims * num_edges, num_dims, num_timesteps]
    edges = torch.cat([receivers, senders], dim=1)
    return edges

  def forward(self, inputs, rel_rec, rel_send):
    # Input now has shape: [num_sims, num_atoms, num_timesteps, num_dims]
    edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
    x = self.cnn(edges)
    x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
    bs = x.shape[0]
    x = self.bn1(self.mlp1(x).reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)
    x_skip = x

    x = self.edge2node(x, rel_rec, rel_send)
    x = self.bn2(self.mlp2(x).reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)

    x = self.node2edge(x, rel_rec, rel_send)
    x = torch.cat((x, x_skip), dim=2)  # Skip connection
    x = self.bn3(self.mlp3(x).reshape(-1,self.n_hid)).reshape(bs,-1,self.n_hid)

    return F.softmax(self.fc_out(x), dim=2)

#
# class CNNStructureEncoder(Encoder):
#   def __init__(self, n_in, n_hid, n_out, do_prob=0.):
#     super(CNNStructureEncoder, self).__init__()
#     self.dropout_prob = do_prob
#
#     self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
#     self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
#     self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
#     self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
#     self.fc_out = nn.Linear(n_hid, n_out)
#
#     print("Using CNN state encoder, hidden size {}.".format(n_hid))
#
#     self.init_weights()
#
#   def node2edge_temporal(self, inputs, rel_rec, rel_send):
#     # NOTE: Assumes that we have the same graph across all samples.
#
#     x = inputs.view(inputs.size(0), inputs.size(1), -1)
#
#     receivers = torch.matmul(rel_rec, x)
#     receivers = receivers.view(inputs.size(0) * receivers.size(1),
#                                inputs.size(2), inputs.size(3))
#     receivers = receivers.transpose(2, 1)
#
#     senders = torch.matmul(rel_send, x)
#     senders = senders.view(inputs.size(0) * senders.size(1),
#                            inputs.size(2),
#                            inputs.size(3))
#     senders = senders.transpose(2, 1)
#
#     # receivers and senders have shape:
#     # [num_sims * num_edges, num_dims, num_timesteps]
#     edges = torch.cat([receivers, senders], dim=1)
#     return edges
#
#   def forward(self, inputs, rel_rec, rel_send):
#     # input has shape: [num_sims, num_edges, num_edge_modules]
#     # edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
#     # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
#
#     x = self.cnn(edges)
#     x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
#     x = self.mlp1(x)
#     x_skip = x
#
#     x = self.edge2node(x, rel_rec, rel_send)
#     x = self.mlp2(x)
#
#     x = self.node2edge(x, rel_rec, rel_send)
#     x = torch.cat((x, x_skip), dim=2)  # Skip connection
#     x = self.mlp3(x)
#
#     return self.fc_out(x)
