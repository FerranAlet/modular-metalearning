from __future__ import print_function
import copy
import numpy as np
import torch
from torch import nn
from composition import Composer
from structure import Structure
from tqdm import tqdm as Tqdm
import json
# import matplotlib.pyplot as plt
import os
import networkx as nx
# from torchviz import make_dot
from scipy.stats import entropy
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)

class HGNN_Composer(Composer):
  def __init__(self, composer, module_list, loss_fn=None, structure={},
      instructions={}):
    super(HGNN_Composer, self).__init__(composer=composer,
        module_list=module_list,
        loss_fn=loss_fn, structure=structure, instructions=instructions)
    self.graph = self.structure['graph']
    #Order must be:
    # a) Nodes
    # b) Edges
    end_nodes = self.structure['end_nodes']
    end_edges = self.structure['end_edges']

    self.node_modules = self.module_list[:end_nodes]
    self.edge_modules = self.module_list[end_nodes:end_edges]
    self.msg_sz = self.structure['msg_sz']
    self.node_dim = self.structure['node_dim']
    self.update_sz = self.structure['update_sz']
    self.visualize = ('visualize' in self.instructions
        and self.instructions['visualize'])

  def visualize_graph(self):
    # import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(list(range(1, self.structure["num_nodes"])))
    for i, edge in enumerate(map(tuple, self.structure['graph']["edges"])):
      type = 'r' if self.structure["edge_idx_inv"][i] else 'b'
      G.add_edge(*edge, color=type, weight=5)

    # edges = G.edges()
    # colors = [G[u][v]['color'] for u, v in edges]
    # weights = [G[u][v]['weight'] for u, v in edges]
    # nx.draw(G, edges=edges, edge_color=colors, width=5, with_labels=True, font_weight='bold')
    # # plt.show()
    # from datetime import datetime

    # plt.savefig('runs/{}.png'.format(datetime.now()))

  def forward_no_weights(self, x):
    nodes = self.inp_to_graph_inp(x)
    # number of steps to predict: -1 because the first one is input
    # self.visualize_graph()
    steps = self.structure['self_regress_steps']
    self.bs = nodes.shape[0]  # batch_size
    self.n = nodes.shape[1]  # number of nodes
    # incoming messages?
    incoming = [torch.zeros(self.bs, self.n, self.msg_sz).cuda() for _ in range(steps + 1)]
    ## Pointing tensors ##
    edge_sources = self.structure['edge_sources']  # module->[list of nodes]
    edge_sinks = self.structure['edge_sinks']  # module->[list of nodes]
    node_idx = self.structure['node_idx']  # module->[list of nodes]
    node_hist = []
    # node_hist.append(nodes.clone())
    #Sinks
    self.structure['sinks_one_hot'] = []
    # import ipdb; ipdb.set_trace()

    # create one-hot feature embedding of edge sinks?
    for sinks in self.structure['edge_sinks']:
      if sinks == []:
        self.structure['sinks_one_hot'].append(None)
        continue
      z = torch.zeros(self.structure['num_nodes'], len(sinks)).cuda()
      aux_idx = torch.cuda.LongTensor(np.array(sinks)).reshape(-1, len(sinks))
      self.structure['sinks_one_hot'].append(z.scatter_(dim=0, index=aux_idx, value=1.).cuda())
      # print('1-hot: ', self.structure['sinks_one_hot'][-1])

    for step in range(steps):
      # Edge modules: concat input from sources and sinks and feed through edge module
      for i, module in enumerate(self.edge_modules):
        if len(edge_sources[i]) == 0: continue
        sources = nodes[:, edge_sources[i], :].clone()
        sinks = nodes[:, edge_sinks[i], :].clone()
        # concat messages from source and sink to form edge state
        inp = torch.cat([sources, sinks], dim=2)
        # feed through edge module
        out = module(inp.view(-1, inp.shape[2])).view(self.bs, -1, self.msg_sz)
        incoming[step] = (incoming[step] + torch.matmul(self.structure['sinks_one_hot'][i], out))
        # for (j, sink) in enumerate(edge_sinks[i]): # TODO: vectorize
        #   incoming[step][:, sink, :] = (incoming[step][:, sink, :].clone() + out[:, j, :])

      # Node modules: pick added messages and old value -> put new
      for (i, module) in enumerate(self.node_modules):
        if len(node_idx[i]) == 0: continue
        msg = incoming[step][:, node_idx[i], :].clone()
        old = nodes[:, node_idx[i], :].clone()
        # Updates node's value; no worries about updating too early since each node only affects itself.
        aux = torch.cat([msg, old], dim=2).view(-1, self.node_dim + self.msg_sz)
        aux = module(aux).view(self.bs, -1, self.update_sz)
        nodes[:, node_idx[i], -self.update_sz:] = nodes[:, node_idx[i], -self.update_sz:].clone() + aux

      node_hist.append(nodes.clone())

    del self.structure['sinks_one_hot']
    # if np.random.rand() < 1./2500:
    #   import pdb; pdb.set_trace()
    if self.visualize:
      return self.graph_out_to_out(node_hist), [_.data.cpu().numpy() for _ in node_hist]
    else: return self.graph_out_to_out(node_hist)

  def inp_to_graph_inp(self, x):
    '''
    Goes from input tensor to tensor in graph space
    '''
    # graph_type = self.graph['type']
    # if 'original_input_shape' in self.structure:
    #   x = x.reshape(tuple([-1]+list(self.structure['original_input_shape'][1:])))
    self.initial_inp = x #maybe clone()?
    if 'self_regress_steps' in self.structure:
      # taking only the first step of every self_regress_step steps as input upon which we will make predictions
      x = x[0::self.structure['self_regress_steps'],:,:].clone()
    else: raise NotImplementedError
    return x

  def graph_out_to_out(self, node_hist):
    '''
    Goes from output in graph space to final output
    '''
    # graph_type = self.graph['type']
    for i in range(len(node_hist)):
      # import ipdb; ipdb.set_trace()
      node_hist[i] = node_hist[i].reshape((node_hist[i].shape[0], -1))
    out = torch.zeros([node_hist[0].shape[0] * len(node_hist), node_hist[0].shape[1]])
    if 'self_regress_steps' in self.structure:
      for i in range(self.structure['self_regress_steps']):
        out[i::self.structure['self_regress_steps']] = node_hist[i].clone()
      # out = out.reshape((out.shape[0], -1))
      # make_dot(out).save('help.dot')
      # print('Saved figure!')
      # import pdb; pdb.set_trace()
      return out[:self.initial_inp.shape[0]-1]
    else:
      raise NotImplementedError
      return node_hist[-1].reshape((node_hist[-1].shape[0], -1))

class HGNN_Structure(Structure):
  def __init__(self, args):
    self.composer_class = HGNN_Composer
    self.composer_abbreviation = 'H_GRAPH'
    #Loads configuration file for graph composer
    self.composer = args.composer
    assert len(self.composer.split('@')) == 2
    [self.composer, self.composer_file] = self.composer.split('@')
    with open(self.composer_file, 'r') as infile:
      self.graph = json.load(infile)
    self.graph['num_nodes'] = len(self.graph['nodes'])
    self.graph['num_edges'] = len(self.graph['edges'])
    self.graph['num_slots'] = self.graph['num_nodes'] + self.graph['num_edges']
    self.loaded_graph = copy.deepcopy(self.graph)
    self.composer = 'gnn'
    self.type_modules = self.graph['type_modules'].split(',')
    self.num_steps = self.graph['num_steps']
    super(HGNN_Structure, self).__init__(args=args)

    self.has_global_variable = False
    self.PosUsage = None
    self.GNNUsage = np.zeros((
      max(len(self.graph['nodes']),len(self.graph['edges'])), self.tot_modules))
    self.StructureParameters = None

  def save_customized_files(self, directory):
    with open(os.path.join(directory, 'graph.json'), 'w') as outfile:
      json.dump(self.loaded_graph, outfile)

  def get_plot_name(self, args, plot_name):
    if args.plot_name.startswith('wf'):
      extra = args.plot_name[2:]
      if len(extra) and extra[0] in ['-','_']: extra = extra[1:]
      plot_name = self.composer_file[:-5] #remove json
      plot_name += 'nm=' + args.num_modules + 'lr='+str(args.adam_lr)
      plot_name += 'opt='+str(args.optimization_steps)
      plot_name += 'data='+args.data_desc.split('@')[1][:5]
      plot_name = plot_name.replace('/','')
      plot_name += 'name='+extra
      plot_name += '/'
      print('Plot name: ', plot_name)
      input('press enter to continue')

  def initialize_all_structures(self, T):
    #Initialize node x node -> edge
    find_node = self.find_node
    if find_node: self.StructureParameters = nn.ParameterList()
    if 'num_types' in self.graph:
      self.num_types = self.graph['num_types']
    else: self.num_types = 2
    C = 5
    self.TypeTypeToEdge = C*np.ones((self.num_types, self.num_types,
      self.num_modules[1]))
    self.TypeTypeNode = C*np.ones((self.num_types, self.num_types,
      self.num_modules[0]))
    assert self.num_modules[0] == 1
    self.TrainStructures = [None for _ in T.MTRAIN]
    self.ValStructures = [None for _ in T.MVAL]
    for i in Tqdm(range(len(self.TrainStructures))):
      self.TrainStructures[i] = self.initialize_structure(
          find_node=find_node, dataset=T.MTRAIN[i])
      self.TrainStructures[i]['original_input_shape'] = (
              T.MTRAIN[i].original_input_shape)
      self.TrainStructures[i]['original_output_shape'] = (
              T.MTRAIN[i].original_output_shape)
    for i in range(len(self.ValStructures)):
      self.ValStructures[i] = self.initialize_structure(
          find_node=find_node, dataset=T.MVAL[i])
      self.ValStructures[i]['original_input_shape'] = (
              T.MVAL[i].original_input_shape)
      self.ValStructures[i]['original_output_shape'] = (
              T.MVAL[i].original_output_shape)

  def edges_from_node_types(self, node_types):
    raise NotImplementedError
    return [self.NodeNodeToEdge[node_types[s]][node_types[t]]
        for (s,t) in self.graph['edges']]

  def initialize_structure(self, find_node=False, dataset=None):
    structure = {}
    structure['graph'] = self.graph
    structure['num_node_modules'] = self.num_modules[0]
    structure['num_edge_modules'] = self.num_modules[1]
    structure['num_nodes'] = len(self.graph['nodes'])
    structure['num_edges'] = len(self.graph['edges'])
    structure['num_steps'] = self.graph['num_steps']
    if 'self_regress' in self.graph and self.graph['self_regress']:
      structure['self_regress'] = True
      structure['self_regress_steps'] = structure['num_steps']
    structure['msg_sz'] = self.graph['msg_sz']
    structure['update_sz'] = self.graph['update_sz']
    structure['node_dim'] = self.graph['node_dim']
    structure['initial_input'] = self.graph['initial_input']
    structure['num_types'] = self.num_types
    ## Limits ##
    structure['end_nodes'] = structure['num_node_modules']
    structure['end_edges'] = (structure['end_nodes'] +
        structure['num_edge_modules'])
    structure['end_final'] = structure['end_edges']
    ## Types ##
    structure['types'] = list(map(int, list(
      np.random.choice(structure['num_types'],len(self.graph['nodes'])))))
    ## Nodes ##
    structure['node_idx_inv'] =  list(map(int, list(
      np.random.choice(structure['num_node_modules'],
          len(self.graph['nodes'])))))#node->mod
    structure['node_idx']     = [[] for _ in
        range(structure['num_node_modules'])]
    for i in range(len(structure['node_idx_inv'])):
      idx = structure['node_idx_inv'][i]
      structure['node_idx'][idx].append(i) #module->[list of nodes]
    ## Edges ##
    structure['edge_idx_inv'] =  list(map(int, list(np.random.choice(
      self.num_modules[1],len(self.graph['edges'])))))#edge->mod

    ## to make sure they are undirected
    # for idx, module in enumerate(structure['edge_idx_inv']):
    #   r, c = tuple(structure['graph']['edges'][idx])
    #   other_idx = structure['graph']['edges'].index([c, r])
    #   structure['edge_idx_inv'][other_idx] = module

    self.update_edge_variables(structure)

    # Find node variables
    if find_node:
       self.StructureParameters.extend([torch.nn.Parameter(
         torch.rand_like(dataset.TrainInput[:,:1,:])*2.-1.)])
       structure['parameters'] = range(len(self.StructureParameters)-1,
           len(self.StructureParameters))
    return structure

  def update_edges_from_nodes_all_structures(self):
    for  i in range(len(self.TrainStructures)):
      self.update_edges_from_nodes(self.TrainStructures[i])
    for  i in range(len(self.ValStructures)):
      self.update_edges_from_nodes(self.ValStructures[i])

  def update_edge_variables(self, structure):
    # structure['edge_idx_inv'] = (
    #     self.edges_from_node_types(structure['node_idx_inv']))
    structure['edge_sources'] = [[] for _ in
        range(structure['num_edge_modules'])]
    structure['edge_sinks']   = [[] for _ in
        range(structure['num_edge_modules'])]
    structure['edge_idx']     = [[] for _ in
        range(structure['num_edge_modules'])]
    structure['node_receives']= [[] for _ in range(structure['num_nodes'])]
    for i in range(len(structure['edge_idx_inv'])):
      idx = structure['edge_idx_inv'][i]
      #module->[list of edge indices]
      structure['edge_idx'][idx].append(i)
      #module->[list of node indices]
      structure['edge_sources'][idx].append(self.graph['edges'][i][0])
      #module->[list of node indices]
      structure['edge_sinks'][idx].append(self.graph['edges'][i][1])
      #'inverse' from edge_sinks
      structure['node_receives'][self.graph['edges'][i][1]].append(i)

  def reset_global_variable(self):
    raise NotImplementedError

  def set_new_global_variable(self):
    '''
    Mutates global variables
    '''
    raise NotImplementedError

  @staticmethod
  def _zero_out_current_probs(curr_edge_idx_inv, probs):
    for edge, module in enumerate(curr_edge_idx_inv):
      probs[edge, module] = 0

  @staticmethod
  def _normalize_probabilities(probs, axis=None):
    dividend = np.sum(probs, axis=axis, keepdims=True)
    np.divide(probs, dividend, out=probs)

  def draw_new_edges_for_node(self, flip=False):
    '''
    given a current structure and an array (of the same dimensions)
    of probabilitys of changing each value, and whether to flip
    or sample, draws new edges for a randomly chosen node
    '''
    def f(new_struct, probs):
      # pick a node to change edges for
      node = np.random.choice(range(new_struct['num_nodes']))
      np_edges_array = np.array(new_struct['graph']['edges'])
      # get first item, because where returns a tuple
      indices_for_node = np.where(np_edges_array[:, 1] == node)[0]
      probabilities = copy.deepcopy(probs)
      if new_struct['num_edge_modules'] > 1:
        if flip:
          self._zero_out_current_probs(new_struct['edge_idx_inv'],
              probabilities)
          self._normalize_probabilities(probabilities, axis=1)
        for idx in indices_for_node:
          new_module = np.random.choice(
              range(new_struct['num_edge_modules']), p=probabilities[idx])
          new_struct['edge_idx_inv'][idx] = new_module
        self.update_edge_variables(new_struct)
      else:
        raise RuntimeError("please check to make sure \
            either node or edge modules > 1")
      return new_struct

    return f

  def draw_new_structure(self, new_struct, probs):
      '''given a current structure and an array (of the same dimensions)
      of probabilities of changing each value, draws a value to change'''
      probabilities = copy.deepcopy(probs)
      self._zero_out_current_probs(new_struct['edge_idx_inv'], probabilities)
      self._normalize_probabilities(probabilities)
      probabilities = probabilities.flatten()

      if new_struct['num_edge_modules'] > 1:
        import itertools
        choices_to_change = list(itertools.product(range(new_struct['num_edges']),
                                                   range(self.num_modules[1]))) # edge, module
        choice_idx = np.random.choice(range(len(choices_to_change)), p=probabilities)
        idx, new_module = choices_to_change[choice_idx] # edge, new module to change to
        new_struct['edge_idx_inv'][idx] = new_module

        # undirected graph: change corresponding edge going the other way
        # r, c = tuple(new_struct['graph']['edges'][idx])
        # other_idx = new_struct['graph']['edges'].index([c, r])
        # new_struct['edge_idx_inv'][other_idx] = new_module

        self.update_edge_variables(new_struct)
      else:
        raise RuntimeError("please check to make sure either node or edge modules > 1")
      return new_struct

  def update_structure_to(self, structure, new_edges):
    structure['edge_idx_inv'] = new_edges
    self.update_edge_variables(structure)

  def propose_new_structure(self, new_structure):
    #Pick either a node or an edge and try switching module
    change_node = (np.random.rand() > 0.5)
    if new_structure['num_node_modules'] == 1: change_node = False
    if new_structure['num_edge_modules'] == 1: change_node = True
    if change_node and new_structure['num_node_modules']>1:
      idx = -1
      while (idx == -1 or #don't modify nodes w/ no assigned module
          new_structure['node_idx_inv'][idx] >=
          new_structure['num_node_modules']):
        idx = np.random.randint(len(new_structure['node_idx_inv']))
      #Remove from old
      old_module = new_structure['node_idx_inv'][idx]
      pos_in_old = new_structure['node_idx'][old_module].index(idx)
      del new_structure['node_idx'][old_module][pos_in_old]
      #Add to new
      new_module = old_module
      while new_module == old_module:
        new_module = np.random.randint(self.num_modules[0])
      new_structure['node_idx_inv'][idx] = new_module
      new_structure['node_idx'][new_module].append(idx)
    elif new_structure['num_edge_modules'] > 1:
      idx = -1
      while (idx == -1 or #don't modify edges w/ no assigned module
          new_structure['edge_idx_inv'][idx] >=
          new_structure['num_edge_modules']):
        idx = np.random.randint(len(new_structure['edge_idx_inv']) // 2)
      #Add to new
      new_module = np.random.randint(self.num_modules[1])
      new_structure['edge_idx_inv'][idx] = new_module

      # undirected graph: change corresponding edge going the other way
      # r, c = tuple(new_structure['graph']['edges'][idx])
      # other_idx = new_structure['graph']['edges'].index([c, r])
      # new_structure['edge_idx_inv'][other_idx] = new_module

      self.update_edge_variables(new_structure)
    else:
      raise RuntimeError("please check to make sure either node or edge modules > 1")
    return new_structure

  def update_PosUsage_counters(self, METRICS):
    '''
    Updates table of fraction of times module is used for each dataset.
    and for each slot.
    '''
    eps = 1./30
    if self.PosUsage is None:
      self.PosUsage = [np.zeros((self.graph['num_slots'], self.tot_modules))
          for _ in range(len(self.TrainStructures)+len(self.ValStructures))]
    for i in range(len(self.PosUsage)):
      self.PosUsage[i] *= (1-eps)
    for i,structure in enumerate(self.TrainStructures+self.ValStructures):
      for j, node in enumerate(structure['node_idx_inv']):
        if node>=0 and node<self.num_modules[0]:
          self.PosUsage[i][j][node] += eps
      for j, node in enumerate(structure['edge_idx_inv']):
        if (node>=self.num_modules[0] and
            node<self.num_modules[0]+self.num_modules[1]):
          self.PosUsage[i][j][node+self.num_modules[0]] += eps
    METRICS['PosUsage'] = [
        [[b.item() for b in a] for a in _] for _ in self.PosUsage]

  def update_customized_counters(self, METRICS=None):
    return #TODO: implement, but not critical, just debugging

  def update_Usage_counters(self, METRICS, T):
    '''
    Updates table of fraction of times module is used for each dataset.
    '''
    eps = 1e-3
    self.Usage *= (1-eps)
    for i_s, structure in enumerate(self.TrainStructures+self.ValStructures):
      for i, l in enumerate(structure['node_idx']):
        self.Usage[i_s][self.Modules[0][i]] += (
            eps * len(l)/structure['num_nodes'])
      for i, l in enumerate(structure['edge_idx']):
        self.Usage[i_s][self.Modules[1][i]] += (
            eps * len(l)/structure['num_edges'])
    names = (
        [_.name for _ in T.MTRAIN] + [_.name for _ in T.MVAL])
    names = list(enumerate(names))
    names.sort(key = lambda x : x[1])
    values = self.Usage[[_[0] for _ in names],:]
    METRICS['Usage'] = [[values[i][j] for j in range(values.shape[1])]
        for i in range(values.shape[0]) ]
    METRICS['Usage-names'] = [_[1] for _ in names]

  def modules_given_structure(self, structure):
    return ([m for m in structure['node_idx_inv']] +
        [int(m)+structure['end_nodes'] for m in structure['edge_idx_inv']])

  def plot_customized_usage_rate(self, directory):
    return #TODO: implement, but not critical, just debugging

  @staticmethod
  def compose_multiple_structures(structures):
    '''
    :return: dictionary representing a mega-graph made of several structure propositions,
     which can be fed into a composer class to be composed into an NN
    '''
    mega_structure = copy.deepcopy(structures[0])
    if len(structures) > 1:
      for structure in structures[1:]:
        # add each structure to mega-composition

        # store for use in calculating updated indices
        prev_num_nodes = mega_structure['num_nodes']
        prev_num_edges = mega_structure['num_edges']

        # num nodes, num edges
        mega_structure['num_nodes'] += structure['num_nodes']
        mega_structure['num_edges'] += structure['num_edges']
        mega_structure['graph']['num_nodes'] += structure['graph']['num_nodes']
        mega_structure['graph']['num_edges'] += structure['graph']['num_edges']

        # create the mega graph
        mega_structure['graph']['nodes'] += structure['graph']['nodes']
        mega_structure['graph']['edges'] += [[node_idx + prev_num_nodes for node_idx in edge]
                                             for edge in structure['graph']['edges']]

        for i, edge_type_list in enumerate(structure['edge_sources']):
            mega_structure['edge_sources'][i] += [node_idx + prev_num_nodes for node_idx in edge_type_list]
        for i, edge_type_list in enumerate(structure['edge_sinks']):
            mega_structure['edge_sinks'][i] += [node_idx + prev_num_nodes for node_idx in edge_type_list]

        # node to type_idx
        for i, node_type_list in enumerate(structure['node_idx']):
            mega_structure['node_idx'][i] += [node_idx + prev_num_nodes for node_idx in node_type_list]
        for i, edge_type_list in enumerate(structure['edge_idx']):
            mega_structure['edge_idx'][i] += [edge_idx + prev_num_edges for edge_idx in edge_type_list]

        # type_list to node_idx
        mega_structure['node_idx_inv'] += structure['node_idx_inv']
        mega_structure['edge_idx_inv'] += structure['edge_idx_inv']

    return mega_structure
