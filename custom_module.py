'''
Heavily inspired by:
https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py
'''
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
from layers import linear, relu, exponential
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)


class torch_NN(nn.Module):
  '''
  Mimic the pytorch-maml/src/ominglot_net.py structure
  '''
  def __init__(self, inp=1, out=1, hidden=[], final_act='affine', loss_fn=None):
    super(torch_NN, self).__init__()
    self.inp = inp
    self.dummy_inp = torch.randn(8, inp, device=nn_device)
    self.out = out
    self.num_layers = len(hidden) + 1
    self.final_act = final_act
    key_words = []
    for i in range(self.num_layers):
      key_words.append('fc'+str(i))
      if i < self.num_layers-1: #final_act may not be a relu
        key_words.append('relu'+str(i))

    def module_from_name(name):
      if self.num_layers >10: raise NotImplementedError
      #TODO: allow more than 10 layers, put '_' and use split
      num = int(name[-1])
      typ = name[:-1]
      if typ == 'fc':
        inp = self.inp if num==0 else hidden[num-1]
        out = self.out if num+1==self.num_layers else hidden[num]
        return nn.Linear(inp, out)
      elif typ=='relu':
        return nn.ReLU() #removed inplace
      else: raise NotImplementedError

    self.add_module('features', nn.Sequential(OrderedDict([
      (name, module_from_name(name)) for name in key_words])))

    if self.final_act == 'sigmoid': self.add_module('fa', nn.Sigmoid())
    elif self.final_act == 'exp': self.add_module('fa', exponential())
    elif self.final_act == 'affine': self.add_module('fa', nn.Sequential())
    elif self.final_act == 'relu': self.add_module('fa', nn.ReLU())
    elif self.final_act == 'tanh': self.add_module('fa', nn.Tanh())
    else: raise NotImplementedError

  def dummy_forward_pass(self):
    '''
    Dummy forward pass to be able to backpropagate to activate gradient hooks
    '''
    return torch.mean(self.forward(self.dummy_inp))

  def forward(self, x, weights=None, prefix=''):
    '''
    Runs the net forward; if weights are None it uses 'self' layers,
    otherwise keeps the structure and uses 'weights' instead.
    '''
    if weights is None:
      x = self.features(x)
      x = self.fa(x)
    else:
      for i in range(self.num_layers):
        x = linear(x, weights[prefix+'fc'+str(i)+'.weight'],
                weights[prefix+'fc'+str(i)+'.bias'])
        if i < self.num_layers-1: x = relu(x)
      x = self.fa(x)
    return x

  def net_forward(self, x, weights=None):
    return self.forward(x, weights)

  #pytorch-maml's init_weights not implemented; no need right now.

  def copy_weights(self, net):
    '''Set this module's weights to be the same as those of 'net' '''
    for m_from, m_to in zip(net.modules(), self.modules()):
      if (isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d)
          or isinstance(m_to, nn.BatchNorm2d)):
        m_to.weight.data = m_from.weight.data.clone()
        if m_to.bias is not None:
            m_to.bias.data = m_from.bias.data.clone()

def main():
  NN = torch_NN(hidden=[2,3], final_act='sigmoid')
  x = torch.rand(1)
  print(NN(x))

if __name__ == '__main__': main()
