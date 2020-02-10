import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 15
import socket, copy, os, time, torch, data_loading
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from encoder import MLPStateEncoder, MLPStructureEncoder, CNNStateEncoder
from custom_module import torch_NN
from data_loading import MetaNpySelfRegressDataset, MetaHDFDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter


class BounceGrad(object):
  def __init__(self, S, args):
    ##############################
    # Parse parameters from args #
    ##############################
    self.S = S
    self.composer = args.composer
    self.torch_seed = args.torch_seed
    torch.manual_seed(self.torch_seed)


    # Device
    if not torch.cuda.is_available() or args.nn_device == 'cpu':
      self.nn_device = 'cpu'
      torch.set_default_tensor_type('torch.FloatTensor')
    else:
      self.nn_device = args.nn_device
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.device(self.nn_device)

    # other
    print('Num threads before: ', torch.get_num_threads())
    if args.cpu_threads > 0: torch.set_num_threads(args.cpu_threads)
    print('Num threads after: ', torch.get_num_threads())
    self.input_noise = args.input_noise
    self.plot_with_without_node = args.plot_with_without_node
    self.stop_edge_acc = args.stop_edge_acc
    self.temporal_acc_SA_truth_mval = {}
    self.auto_temp = args.auto_temp
    self.S.find_node = args.find_node
    self.repeat_mtrain = args.repeat_mtrain
    self.normalize_output = args.normalize_output
    self.normalize_data = args.normalize_data
    self.split_by_file = args.split_by_file
    self.limit_data = args.limit_data
    self.do_bounce = args.do_bounce
    self.encoder_lr = args.encoder_lr
    self.dont_accept = args.dont_accept
    self.smaller_MVals = (list(map(int, args.smaller_MVals.split(',')))
                          if args.smaller_MVals != '' else [])
    self.meta_split = [x / 100 for x in map(int, args.meta_split.split(','))]

    self.data_split = [x / 100 for x in map(int, args.data_split.split(','))]
    self.initial_temp = args.initial_temp
    self.min_temp = np.exp(args.min_temp)
    self.initial_acc = args.initial_acc
    self.meta_lr = args.meta_lr
    self.meta_batch_size = args.meta_batch_size
    self.optimization_steps = args.optimization_steps
    self.temp_slope_opt_steps = args.temp_slope_opt_steps
    if self.temp_slope_opt_steps == -1:
      self.temp_slope_opt_steps = self.optimization_steps
    self.max_datasets = args.max_datasets
    self.do_mtrain = args.do_mtrain

    self.data_desc = args.data_desc
    self.initialize_dataset(self.data_desc)
    # initialize encoder, everys loss, etc.
    self.everys = [1, 10, 20]
    self.proposal_type = args.proposal_type
    self.proposal_flip = args.proposal_flip
    self.encoder_input_type = args.encoder_input_type
    if self.encoder_input_type not in ["truth", "none"]:
      self.encoder_hs = args.encoder_hs
      self.initialize_encoder(self.encoder_input_type, args.encoder_type, self.encoder_hs)
      self.enc_train_only_acc = args.enc_train_only_acc
      # proposal method
      if self.proposal_type == "node":
        self.proposal_fn = self.S.draw_new_edges_for_node(self.proposal_flip)
      elif self.proposal_type == "edge":
        self.proposal_fn = self.S.draw_new_structure
      else:
        raise RuntimeError("should not get here")
    if self.encoder_input_type in ["structure", "both"]:
      self.structure_enc_input = args.structure_enc_input

    # load dataset and modules
    self.load_modules = args.load_modules
    self.load_structures_and_metrics = args.load_structures_and_metrics
    self.initialize_modules()

    # Metrics and modules checkpoint
    self.log_dir = "{}/{}".format(args.log_dir, self.get_filename(args.log_dir_comment))
    print("saving metrics to:", self.log_dir)
    self.writer = SummaryWriter(log_dir=self.log_dir)
    self.save_modules_every = args.plot_freq
    self.models_path = os.path.join(args.models_path,
            self.get_filename(args.log_dir_comment))

  def get_filename(self, comment=''):
    hostname = socket.gethostname()
    fmt_time = time.strftime('%b%d_%H-%M-%S')
    if not hasattr(self.S, 'num_steps'): self.S.num_steps = None
    suffix = "sim={}_mbs={}_ds={}_dspl={}_os={}_lr={}_it={}_ia={}_ns={}_enc={}{}{}{}{}{}_{}".format(
      self.data_source.split('@')[1], self.meta_batch_size, self.max_datasets,
      ",".join(map(str, self.data_split)),
      self.optimization_steps, self.meta_lr, self.initial_temp,
      self.initial_acc, self.S.num_steps, self.encoder_input_type,
      "_prptyp={}".format(self.proposal_type) if self.encoder_input_type not in ["truth", "none"] else "",
      "_prpflp={}".format(self.proposal_flip) if self.proposal_type == "node" else "",
      "_ehs={}".format(self.encoder_hs) if self.encoder_input_type not in ["truth", "none"] else "",
      "_encin={}".format(self.structure_enc_input) if self.encoder_input_type == "structure" else "",
      "_toa={}".format(self.enc_train_only_acc) if self.encoder_input_type not in ["truth", "none"] else "",
      '_' + comment)
    filename = "{}_{}_{}".format(fmt_time, hostname, suffix)
    return filename

  @staticmethod
  def _encode_onehot(array, num_classes):
    """
    :param labels: array of idx -> edge_module (structure['edge_idx_inv'])
    :return: numpy one-hot encoding where dim 0 is edge and dim 1 is one-hot encoding
    """
    assert num_classes >= len(set(array))
    num_edges = len(array)
    one_hot = np.zeros((num_edges, num_classes))
    one_hot[np.arange(num_edges), array] = 1
    return one_hot

  def initialize_encoder(self, encoder_input_type, encoder_type, encoder_hs):
    if encoder_input_type == "state" and encoder_type == "mlp":
      self.encoder = MLPStateEncoder(49 * 4, encoder_hs, 2, 0.5)
    elif encoder_input_type == "state" and encoder_type == "cnn":
      self.encoder = CNNStateEncoder(4, encoder_hs, 2, 0.5)
    elif encoder_input_type == "structure" and encoder_type == 'mlp':
      self.encoder = MLPStructureEncoder(2, encoder_hs, 2, 0.5)
    elif encoder_input_type == "both":
      self.encoder = MLPStateEncoder(49 * 4, encoder_hs, 2, 0.5)
      self.encoder2 = MLPStructureEncoder(2, encoder_hs, 2, 0.5)
    elif encoder_input_type == "truth":
      return
    else:
      raise RuntimeError("should not get here", encoder_input_type,
              encoder_type, encoder_hs)

    num_nodes = self.S.graph['num_nodes']
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(self._encode_onehot(np.where(off_diag)[1], num_nodes), dtype=np.float32)
    rel_send = np.array(self._encode_onehot(np.where(off_diag)[0], num_nodes), dtype=np.float32)
    self.rel_rec = torch.Tensor(rel_rec)
    self.rel_send = torch.Tensor(rel_send)

    self.encoder_loss_fn = nn.BCELoss(reduction='mean')
    self.encoder_opt = optim.SGD(self.encoder.parameters(),
        lr=self.encoder_lr, momentum=0.9)

    if encoder_input_type == "both":
      self.encoder_loss_fn2 = nn.BCELoss(reduction='mean')
      self.encoder_opt2 = optim.SGD(self.encoder2.parameters(), lr=0.001, momentum=0.9)

  def initialize_modules(self):
    '''
    Creates modules following num_modules and type_modules
    '''
    self.L = nn.ModuleList()  # Library of PyTorch Modules
    self.S.Modules = []
    self.nn_inp = []
    self.nn_out = []
    self.nn_hid = []
    self.nn_act = []
    for (t, (num, typ)) in enumerate(zip(self.S.num_modules, self.S.type_modules)):
      l = []
      # 'final_act'-#inp-#hid1-#hid2-...-#out
      # example: 'affine-128-64-64-42'
      typ_split = typ.split('-')
      for i in range(len(typ_split)):
        try:
          typ_split[i] = int(typ_split[i])
        except: pass
      self.nn_act.append(typ_split[0])
      self.nn_inp.append(typ_split[1])
      self.nn_hid.append(typ_split[2:-1])
      self.nn_out.append(typ_split[-1])
      for _ in range(num):
        aux_nn = torch_NN(inp=self.nn_inp[t], out=self.nn_out[t],
          hidden=self.nn_hid[t],
          final_act=self.nn_act[t]).to(device=self.nn_device)
        l.append(len(self.L))
        self.L.append(aux_nn)
      self.S.Modules.append(l)
    if self.load_modules != '': self.load_L(self.load_modules)
    self.SOpt = torch.optim.Adam(self.L.parameters(), lr=self.meta_lr)
    self.SOpt_scheduler = ReduceLROnPlateau(
      optimizer=self.SOpt, factor=1 / 2.,
      mode='min', patience=self.optimization_steps/100, threshold=0,
      cooldown=self.optimization_steps/20, verbose=True, min_lr=1e-5)
    self.LocalOpt = None  # optimizer of customized parameters --> trained w/ Train, not Val
    self.initial_norms = [torch.norm(_) for m in self.L for _ in m.parameters()]

  def initialize_dataset(self, data_desc):
    '''
    Creates dataset by calling dataset.py and transforms it to pytorch tensors.
    '''
    # Parsing
    self.parsed_data_desc = data_desc.split('-')
    self.data_source = self.parsed_data_desc[0]
    if self.data_source.startswith('HDF5'):
      self.D = MetaHDFDataset(
        mtrain=self.meta_split[0], mval=self.meta_split[1],
        mtest=self.meta_split[2],
        train=self.data_split[0], val=self.data_split[1],
        test=self.data_split[2],
        filename=self.data_source.split('@')[1],
        limit_data=self.limit_data,
        max_datasets=self.max_datasets,
        #split_by_file=self.split_by_file,
        smaller_MVals=self.smaller_MVals,
        normalize_output=self.normalize_output,
        normalize=self.normalize_data)
      self.function_depth = 2
    elif self.data_source.startswith('NPY_SR'):
      self.D = MetaNpySelfRegressDataset(
        # mtrain=self.meta_split[0], mval=self.meta_split[1],
        # mtest=self.meta_split[2],
        train=self.data_split[0], val=self.data_split[1],
        test=self.data_split[2],
        filename=self.data_source.split('@')[1],
        limit_data=self.limit_data,
        max_datasets=self.max_datasets,
        # split_by_file=self.split_by_file,
        smaller_MVals=self.smaller_MVals,
        normalize=self.normalize_data)
    else:
      raise RuntimeError('{} hasnt been implemented yet'.format(self.data_source))

    self.T = data_loading.convert_to_torch_tensors(self.D, self.nn_device)
    if 'cars' in self.data_desc:
      for i in range(len(self.T.MTRAIN)):
        self.T.MTRAIN[i].TrainInput = self.T.MTRAIN[i].TrainInput[:50]
        self.T.MTRAIN[i].TrainOutput = self.T.MTRAIN[i].TrainOutput[:49]
        self.T.MTRAIN[i].ValInput = self.T.MTRAIN[i].ValInput[:50]
        self.T.MTRAIN[i].ValOutput = self.T.MTRAIN[i].ValOutput[:49]
    if self.plot_with_without_node:
      self.DL_Mtrain, self.DL_Mval, self.DL_Mtest \
        = data_loading.get_data_loaders(self.T,
            batch_size=self.meta_batch_size, shuffle=False)
    else:
      self.DL_Mtrain, self.DL_Mval, self.DL_Mtest \
        = data_loading.get_data_loaders(self.T, batch_size=self.meta_batch_size)

    self.initialize_answers()

  #######################################
  ###                                 ###
  ### BOUNCEGRAD: SAnnealing with SGD ###
  ###                                 ###
  #######################################

  def run_model(self, structure, inp, instructions={}):
    '''
    Returns the composition of several modules
    '''
    if len(inp.shape) == 0 or inp.shape[0] == 0: return inp  # empty tensor
    if len(inp.shape) == 1: inp = inp.unsqueeze(1)
    return self.S.composer_class(composer=self.composer,
        module_list=self.L, structure=structure,instructions=instructions)(inp)

  def batched_evaluate(self, structures, datasets, mode):
    '''
    Evaluates the dataset according to the structure
    '''
    mega_structure = self.S.compose_multiple_structures(structures)
    num_nodes = structures[0]["original_input_shape"][1]

    if self.S.find_node:
      #this will only work if ValInput == TrainInput == shifted TrainOutput
      #which is satisfied for the find_node experiments.
      inp = torch.cat([torch.cat(
        [self.S.StructureParameters[structure['parameters'][0]],
          dataset.TrainInput[:,1:,:]], dim=1)
        for (structure, dataset) in zip(structures, datasets)],
        dim=1).contiguous()
      param_two = self.S.StructureParameters

      inp_two = torch.cat([torch.cat([p_two, dataset.TrainInput[:,1:,:]], dim=1)
        for (p_two, structure, dataset) in zip(param_two,structures, datasets)],
        dim=1).contiguous()
      mega_structure['self_regress_steps'] = 50
      preds_two = self.run_model(mega_structure, inp_two).reshape(-1, *inp.shape[1:])
      preds_two = list(torch.split(preds_two, num_nodes, dim=1))
      preds = self.run_model(mega_structure, inp).reshape(-1, *inp.shape[1:])
      preds = list(torch.split(preds, num_nodes, dim=1))
      seen_outs = [dataset.TrainOutput[1:] for dataset in datasets]
      for i, (struct, pred, pred_two, out) in enumerate(zip(structures, preds,
          preds_two, seen_outs)):
        if F.mse_loss(pred[1:], out) > F.mse_loss(pred_two[1:], out):
          # New random proposal is better
          self.S.StructureParameters[struct['parameters'][0]].data = \
              param_two[i].data
          preds[i] = preds_two[i]
        struct['find_node_preds'] = torch.cat([
          self.S.StructureParameters[struct['parameters'][0]][:1],
          preds[i][:,:1,:]], dim=0).detach().cpu().numpy()

      inp = torch.cat([torch.cat(
        [self.S.StructureParameters[structure['parameters'][0]],
          dataset.TrainInput[:,1:,:]], dim=1)
        for (structure, dataset) in zip(structures, datasets)],
        dim=1).contiguous()
      # mega_structure['self_regress_steps'] = 10
      preds = self.run_model(mega_structure, inp).reshape(-1, *inp.shape[1:])
      preds = list(torch.split(preds, num_nodes, dim=1))
      unseen_objs = [torch.cat(
          [self.S.StructureParameters[structure['parameters'][0]][:1],
            pred[:,:1,:]], dim=0)
          for (pred, structure, dataset) in zip(preds, structures, datasets)]
      inp = torch.cat([torch.cat([
        unseen_obj, dataset.TrainInput[:,1:,:]], dim=1)
        for (structure, dataset, unseen_obj) in
        zip(structures, datasets, unseen_objs)], dim=1).contiguous()
      outs = [torch.cat([
        unseen_obj[1:], dataset.TrainOutput[:,1:,:]], dim=1).contiguous()
        for (structure, dataset, unseen_obj)
        in zip(structures, datasets, unseen_objs)]
      true_inp = torch.cat([dataset.TrainInput for dataset in datasets], dim=1)
      for i in range(inp.shape[0]):
        self.find_node_dist_rels[
            np.sum(datasets[i].Edges[:4]).item()].append(
            torch.mean((inp[:,5*i:5*i+1]-
              true_inp[:,5*i:5*i+1])**2).detach().cpu().numpy().item())
      self.S.find_node_dist += (torch.mean((inp[:,np.arange(0,inp.shape[1],5)]-
        true_inp[:,np.arange(0,inp.shape[1],5)])**2
        ).detach().cpu().numpy().item())
      self.find_ct += 1.
    else:
      inp = torch.cat([dataset.TrainInput if mode == 'Train'
                 else dataset.ValInput
                 for dataset in datasets], dim=1)
      outs = [dataset.TrainOutput if mode == 'Train'
             else dataset.ValOutput
             for dataset in datasets]
    if len(inp.shape) == 0 or inp.shape[0] == 0:  # empty tensors
      return torch.FloatTensor(np.array([0])), outs

    if self.BG_mode == 'train' and self.input_noise > 0.:
      inp = inp + torch.randn_like(inp)*self.input_noise
    preds = self.run_model(mega_structure, inp)
    preds = preds.reshape(-1, *inp.shape[1:])
    preds = torch.split(preds, num_nodes, dim=1)
    preds_every = [[pred[num_steps - 1,:,:] for num_steps in self.everys]
        for pred in preds]
    outs_every = [[out[num_steps - 1,:,:] for num_steps in self.everys]
        for out in outs]
    # inps_every = [[inpp[num_steps - 1,:,:] for num_steps in self.everys]
    #     for inpp in inps]

    # Compute residual correlations
    residuals = torch.cat([pred-out for pred, out in zip(preds, outs)],
        dim=1).detach()
    before = residuals[:-1] ; after = residuals[1:]
    before = before-torch.mean(before) ; after = after-torch.mean(after)
    self.residual_autocorr = (torch.sum(before*after)/
        (torch.norm(before)*torch.norm(after))).detach().cpu().numpy().item()

    return [F.mse_loss(pred, out) for pred, out in zip(preds, outs)], \
           [[F.mse_loss(pred, out) for pred, out in zip(p, o)]
            for p, o in zip(preds_every, outs_every)], \
           preds

  def _update_frac_worse_accepts(self, old_loss, new_loss, upt_factor, accept):
    if old_loss < new_loss:
      y = upt_factor if accept else 0
      self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
      self.SA_running_acc_rate = ((1 - upt_factor) * self.SA_running_acc_rate + y)

  def _get_structure_encoder_input(self, structures):
    '''
    :param list of structures input to encoder
    :return: torch tensor input to encoder
    '''
    def select_indices(array, proportion):
      num_edges = array.shape[0]
      num_to_choose = round(proportion * num_edges)
      indices = np.random.choice(num_edges, num_to_choose, replace=False)
      return indices

    def uniformize_some(array, proportion):
      """
      :param array: one-hot encoding of idx -> edge_module (structure['edge_idx_inv'])
      :param proportion: proportion of edges to uniformize
      :return: nearly one-hot encoding where proportion of the edges are uniformized
      to 1/num_modules probability
      """
      array = array.astype("float64")
      indices = select_indices(array, proportion)
      num_modules = array.shape[1]
      uniform_value = 1 / num_modules
      array[indices] = np.full(num_modules, uniform_value)
      return array

    def flip_some(array, proportion):
      indices = select_indices(array, proportion)
      values = 1 - array[indices]
      array[indices] = values
      return array

    self.proportion_uniformized = .2

    inps = []
    for structure in structures:
      one_hot = self._encode_onehot(
          structure['edge_idx_inv'], structure['num_edge_modules'])
      if self.structure_enc_input == "flip":
        constructed_inp = flip_some(one_hot, self.proportion_uniformized)
      elif self.structure_enc_input == "uniform":
        constructed_inp = uniformize_some(one_hot, self.proportion_uniformized)
      else:
        raise RuntimeError("should not get here")
      inps.append(torch.Tensor(constructed_inp))

    inp = torch.stack(inps)
    return inp

  def batched_propose(self, encoder, structures, datasets, type="state"):
    proposed_structs = [copy.deepcopy(structure) for structure in structures]
    if type == "state":
      inp = torch.stack([dataset.TrainInput for dataset in datasets], dim=0)
      inp = torch.transpose(inp, 1, 2).contiguous()
    elif type == "structure":
      inp = self._get_structure_encoder_input(structures)
    else:
      raise RuntimeError("should not get here")
    # [num_sims, num_atoms, num_timesteps*num_dims]
    pred_probs = encoder(inp, self.rel_rec, self.rel_send)
    pred_probs_np = pred_probs.detach().cpu().numpy()

    for struct, pred_prob in zip(proposed_structs, pred_probs_np):
      self.proposal_fn(struct, pred_prob)

    return proposed_structs, pred_probs, pred_probs_np

  @staticmethod
  def bidirectional_accuracy(batch_structures):
    def get_other_edge(structure, idx):
      r, c = tuple(structure['graph']['edges'][idx])
      other_idx = structure['graph']['edges'].index([c, r])
      return structure['edge_idx_inv'][other_idx]
    num_bi_correct = np.sum([np.sum([edge == get_other_edge(structure, idx) \
      for idx, edge in enumerate(structure['edge_idx_inv'])]) / len(
      structure['edge_idx_inv']) for structure in batch_structures])
    return num_bi_correct

  def batched_bounce(self, batch_structures, batch_datasets, temp, do_grad):
    '''
    Propose a modification to structure across all datasets
    Do a SA step depending on performance in Train
    Backpropagate on Val
    '''

    ##########################################
    # Propose new structure for each dataset #
    ##########################################
    chosen = None
    if self.encoder_input_type == "both":
      if np.random.uniform() < .5:
        chosen = "state"
        proposed_structs, pred_probs, pred_probs_np = self.batched_propose(
            self.encoder, batch_structures, batch_datasets, chosen)
      else:
        chosen = "structure"
        proposed_structs, pred_probs, pred_probs_np = self.batched_propose(
            self.encoder2, batch_structures, batch_datasets, chosen)
    # propose new structure for each dataset, randomly or give truth
    elif self.encoder_input_type in ["none", "truth"]:
      proposed_structs = []
      for structure, dataset in zip(batch_structures, batch_datasets):
        new_structure = copy.deepcopy(structure)
        if self.encoder_input_type == "none":
          self.S.propose_new_structure(new_structure)
        elif self.encoder_input_type == "truth":
          self.S.update_structure_to(new_structure, dataset.Edges.tolist())
        proposed_structs.append(new_structure)

    # use encoder to propose new structure for each dataset
    elif self.encoder_input_type in ["state", "structure"]:
      proposed_structs, pred_probs, pred_probs_np = \
          self.batched_propose(self.encoder, batch_structures,
              batch_datasets, self.encoder_input_type)

    else:
      raise RuntimeError("should not get here")
    if self.encoder_input_type not in ["truth"]:
      # evaluate old structures
      # [loss], [pred] (list of losses and list of predictions)
      orig_train_loss, orig_train_loss_every, orig_train_pred = \
          self.batched_evaluate(batch_structures, batch_datasets, 'Train')

      if batch_datasets[0].TrainInput.shape != batch_datasets[0].ValInput.shape \
         or torch.any(batch_datasets[0].TrainInput != batch_datasets[0].ValInput):
        orig_val_loss, orig_val_loss_every, orig_val_pred = \
            self.batched_evaluate(batch_structures, batch_datasets, 'Val')
      else:
        orig_val_loss, orig_val_loss_every, orig_val_pred = \
            orig_train_loss, orig_train_loss_every, orig_train_pred
      orig_train_loss_np = map(lambda x: x.data.cpu().numpy(), orig_train_loss)

    ###########################
    # Evaluate new structures #
    ###########################
    new_train_loss, new_train_loss_every, new_train_pred = \
        self.batched_evaluate(proposed_structs, batch_datasets, 'Train')

    if batch_datasets[0].TrainInput.shape != batch_datasets[0].ValInput.shape \
       or torch.any(batch_datasets[0].TrainInput != batch_datasets[0].ValInput):
      new_val_loss, new_val_loss_every, new_val_pred = self.batched_evaluate(
          proposed_structs, batch_datasets, 'Val')
    else:
      new_val_loss, new_val_loss_every, new_val_pred = \
          new_train_loss, new_train_loss_every, new_train_pred
    new_train_loss_np = map(lambda x: x.data.cpu().numpy(), new_train_loss)

    if self.encoder_input_type in ["truth"]:
      train_loss_every, val_loss_every = (
              list(zip(*new_train_loss_every)), list(zip(*new_val_loss_every)))
      train_loss_every = [[p.data.cpu().numpy() for p in e]
              for e in train_loss_every]
      val_loss_every = [[p.data.cpu().numpy() for p in e]
              for e in val_loss_every]
      if do_grad:
        agg_loss = torch.sum(torch.stack(new_val_loss))
        agg_loss.backward()
      new_val_loss_np = [x.data.cpu().numpy() for x in new_val_loss]
      return proposed_structs, new_train_loss_np, new_val_loss_np, \
          train_loss_every, val_loss_every, \
          np.array(0), np.array(0), 0, 0, 0, \
          np.array(0), None, [], np.array(0), 0, None

    ###################################################################
    # Decide whether to accept or reject each structure independently #
    ###################################################################
    return_tuples = []
    accepted_structs = []
    accepted_pred_probs = []
    upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
    for i, (old_loss, new_loss) in enumerate(
            zip(orig_train_loss_np, new_train_loss_np)):
      prob_accept = np.exp((old_loss - new_loss) / temp)
      accept = new_loss <= old_loss or np.random.rand() < prob_accept
      self._update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept)

      if accept and not(self.dont_accept):
        return_tuples.append((proposed_structs[i], new_train_loss[i],
            new_train_loss_every[i], new_train_pred[i], new_val_loss[i],
            new_val_loss_every[i], new_val_pred[i]))
        if self.encoder_input_type not in ["none", "truth"]:
          accepted_structs.append(proposed_structs[i])
          accepted_pred_probs.append(pred_probs[i])
      else: # Reject
        return_tuples.append((batch_structures[i], orig_train_loss[i],
            orig_train_loss_every[i], orig_train_pred[i], orig_val_loss[i],
            orig_val_loss_every[i], orig_val_pred[i]))

    # accepted structure proposal accuracy
    num_edges_correct = sum([float(np.sum(dataset.Edges == np.array(
      structure['edge_idx_inv'])))/dataset.Edges.shape[0]
      for structure, dataset in zip(batch_structures, batch_datasets)])

    # bidirectional accuracy: proportion of all edges that are bidirectional
    num_bi_corr = self.bidirectional_accuracy(batch_structures)

    # proportion of correct edges that are also bidirectional
    # np.array(structure['edge_idx_inv'])[
    #     np.where(dataset.Edges == np.array(structure['edge_idx_inv']), )]

    if self.encoder_input_type not in ["none", "truth"]:
      # encoder predicted proposal accuracy
      batch_preds = list(map(lambda x: x.squeeze(0),
        np.split(np.where(pred_probs_np[:,:,1] > .5, 1, 0),
          pred_probs.shape[0], axis=0)))
      encoder_num_edges_correct = sum([
        float(np.sum(dataset.Edges == pred)) / dataset.Edges.shape[0]
        for pred, dataset in zip(batch_preds, batch_datasets)])
      encoder_num_edges_proposal = sum([float(np.sum(
        np.array(structure['edge_idx_inv']) == pred)) / pred.shape[0]
        for pred, structure in zip(batch_preds, batch_structures)])

      # accepted_structs = [dataset.Edges for dataset in batch_datasets] #HACK

    elif self.encoder_input_type in ["none", "truth"]:
      pred_probs = torch.Tensor([0])
      encoder_num_edges_correct = 0
      encoder_num_edges_proposal = 0

    # aggregating the selected structs/loss/preds into lists
    updated_structs, train_loss, train_loss_every, train_pred, \
    val_loss, val_loss_every, val_pred = zip(*return_tuples)

    if do_grad:
      agg_loss = torch.sum(torch.stack(val_loss))
      agg_loss.backward()
    # if self.LocalOpt is not None:
    #   agg_train_loss = torch.sum(torch.stack(train_loss))
    #   agg_train_loss.backward()

    # convert to numpy and zip
    (train_loss, val_loss, train_pred, val_pred) = map(
        lambda x: [p.data.cpu().numpy() for p in x],
        (train_loss, val_loss, train_pred, val_pred))

    train_loss_every, val_loss_every = list(
        zip(*train_loss_every)), list(zip(*val_loss_every))
    train_loss_every = [[p.data.cpu().numpy() for p in e]
        for e in train_loss_every]
    val_loss_every = [[p.data.cpu().numpy() for p in e] for e in val_loss_every]

    if (len(accepted_pred_probs) > 0):
      accepted_pred_probs = torch.stack(accepted_pred_probs)
    else:
      accepted_pred_probs = torch.Tensor(0)

    return updated_structs, train_loss, val_loss, train_loss_every, \
        val_loss_every, train_pred, val_pred, num_edges_correct, \
        encoder_num_edges_correct, encoder_num_edges_proposal, pred_probs, \
        chosen, accepted_structs, accepted_pred_probs, num_bi_corr, None

  def encoder_loss(self, updated_structures, pred_probs, batch_size):
    encoder_labels = [structure["edge_idx_inv"]
        for structure in updated_structures]
    # encoder_labels = updated_structures
    encoder_labels = torch.Tensor(encoder_labels)
    encoder_loss = self.encoder_loss_fn(pred_probs[:, :, 1], encoder_labels)
    encoder_loss = encoder_loss * len(updated_structures) / batch_size
    return encoder_loss

  def encoder_step(self, encoder_loss, chosen):
    if chosen is None:
      self.encoder_opt.zero_grad()
      encoder_loss.backward()
      self.encoder_opt.step()
      self.encoder_opt.zero_grad()
    elif chosen == "structure" or chosen == "state":
      self.encoder_opt2.zero_grad()
      encoder_loss.backward()
      self.encoder_opt2.step()
      self.encoder_opt2.zero_grad()
    else:
      raise RuntimeError("should not get here")

  def do_plot_with_without_node(self, structures, datasets):
    # colors = ['r', 'g', 'b', 'k', 'c']
    num_nodes = datasets[0].TrainInput.shape[1]
    c_a = np.tile(np.array([[0,0.,0.]]), (50,1))
    c_b = np.tile(np.array([[0,0.,1.]]), (50,1))
    c_c = np.tile(np.array([[0,1.,0.]]), (50,1))
    c_d = np.tile(np.array([[0,0.5,0.5]]), (50,1))
    c_e = np.tile(np.array([[0,0.25,0.25]]), (50,1))
    c_f = np.tile(np.array([[0,0.25,0.25]]), (50,1))
    c_g = np.tile(np.array([[0,0.25,0.25]]), (50,1))
    c_h = np.tile(np.array([[0,0.25,0.25]]), (50,1))
    for i in range(50):
      c_a[i,0] = c_b[i,0] = c_c[i,0] = c_d[i,0] = float(i)/50
      c_e[i,0] = c_f[i,0] = c_g[i,0] = c_h[i,0] = float(i)/50
    colors = [c_a,c_b,c_c,c_d,c_e,c_f,c_g,c_h]
    mega_structure = self.S.compose_multiple_structures(structures)
    mega_structure['self_regress_steps'] = 50
    inp = torch.cat([dataset.TrainInput for dataset in datasets], dim=1)
    pred = self.run_model(mega_structure, inp).reshape(-1, *inp.shape[1:])
    preds = list(torch.split(pred, 8, dim=1))
    for (structure, dataset, pred) in zip(structures, datasets, preds):
      fig = plt.figure()
      for j in range(1,num_nodes):
        plt.scatter(dataset.TrainInput[:,j,0].detach().cpu().numpy(),
            dataset.TrainInput[:,j,1].detach().cpu().numpy(),
            s=rcParams['lines.markersize']**2 * 4,
            # alpha=k/dataset.TrainInput.shape[0]/2.,
            c=colors[j][:dataset.TrainInput.shape[0]])
        # for k in range(dataset.TrainInput.shape[0]):
        #   plt.scatter(dataset.TrainInput[k,j,0].detach().cpu().numpy(),
        #       dataset.TrainInput[k,j,1].detach().cpu().numpy(),
        #       s=rcParams['lines.markersize']**2 * 4,
        #       # alpha=k/dataset.TrainInput.shape[0]/2.,
        #       c=colors[j][k])
      plt.xlim([-1,1])
      plt.ylim([-1,1])
      self.writer.add_figure('remove_first_'+str(dataset.structure_idx),
          fig, self.step)
      fig = plt.figure()
      for j in range(num_nodes):
        plt.scatter(dataset.TrainInput[:,j,0].detach().cpu().numpy(),
            dataset.TrainInput[:,j,1].detach().cpu().numpy(),
            s=rcParams['lines.markersize']**2 * 4,
            # alpha=k/dataset.TrainInput.shape[0]/2.,
            c=colors[j][:dataset.TrainInput.shape[0]])
        # for k in range(dataset.TrainInput.shape[0]):
        #   plt.scatter(dataset.TrainInput[k,j,0].detach().cpu().numpy(),
        #       dataset.TrainInput[k,j,1].detach().cpu().numpy(),
        #       s=rcParams['lines.markersize']**2 * 4,
        #       # alpha=k/dataset.TrainInput.shape[0]/2.,
        #       c=colors[j][k])
      self.writer.add_figure('all_nodes_'+str(dataset.structure_idx),
          fig, self.step)
      fig = plt.figure()
      for j in range(num_nodes):
        plt.scatter(dataset.TrainInput[:,j,0].detach().cpu().numpy(),
            dataset.TrainInput[:,j,1].detach().cpu().numpy(),
            s=rcParams['lines.markersize']**2 * 4,
            # alpha=k/dataset.TrainInput.shape[0]/2.,
            c=colors[j][:dataset.TrainInput.shape[0]])
        plt.scatter(pred[:,j,0].detach().cpu().numpy(),
            pred[:,j,1].detach().cpu().numpy(),
            marker='*',
            s=rcParams['lines.markersize']**2 * 4,
            c=colors[j][:pred.shape[0]])
      self.writer.add_figure('comparison_'+str(dataset.structure_idx),
          fig, self.step)
        # for k in range(dataset.TrainInput.shape[0]):
        #   plt.scatter(pred[k,j,0].detach().cpu().numpy(),
        #       pred[k,j,1].detach().cpu().numpy(),
        #       marker='*',
        #       s=rcParams['lines.markersize']**2 * 4,
        #       alpha=k/dataset.TrainInput.shape[0]/2.,
        #       c=colors[j][k])
      # Remove connections to first node for last plot
      for pos in [0,1,2,3,4,8,12,16]:
        structure['edge_idx_inv'][pos] = 0
      # self.S.update_edge_variables(structure)

    pred = self.run_model(mega_structure, inp).reshape(-1, *inp.shape[1:])
    preds = list(torch.split(pred, num_nodes, dim=1))
    for i, (structure, dataset) in enumerate(zip(structures, datasets)):
      fig = plt.figure()
      for j in range(1, num_nodes):
        plt.scatter(preds[i][:,j,0].detach().cpu().numpy(),
            preds[i][:,j,1].detach().cpu().numpy(),
            s=rcParams['lines.markersize']**2 * 4,
            c=colors[j][:pred.shape[0]])
        # for k in range(preds[i].shape[0]):
        #   plt.scatter(preds[i][k,j,0].detach().cpu().numpy(),
        #       preds[i][k,j,1].detach().cpu().numpy(),
        #       s=rcParams['lines.markersize']**2 * 4,
        #       alpha=k/dataset.TrainInput.shape[0]/2.,
        #       c=colors[j][:pred.shape[0]])
      plt.xlim([-1,1])
      plt.ylim([-1,1])
      self.writer.add_figure('counterfactual_'+str(dataset.structure_idx),
          fig, self.step)



  ##############################
  ## MAIN BOUNCEGRAD FUNCTION ##
  ##############################
  def SAConfig_SGDModules(self, optimization_steps):
    '''
    Optimization by Simulated Annealing on the module configurations
    and SGD (Adam in our case) on the modules at every step.
    '''
    # Cooling schedule decreases exponentially wrt the fraction of accepted
    # proposals with worse performance, starting at 1 (accept no matter what)
    # Therefore we have to keep a running estimate of fraction of accepts.
    self.SA_running_acc_rate = 1e-9  # initial counters for Simulated Annealing
    self.SA_running_factor = 1e-9  # normalizing constant
    temp = np.exp(self.initial_temp)  # temperature in the SA formula
    temp_change = 1.1
    self.CosDist = nn.CosineSimilarity(dim=1)

    #############################
    # Create initial structures #
    #############################
    self.S.initialize_all_structures(T=self.T)
    if self.S.StructureParameters is not None and len(self.S.StructureParameters)>0:
      self.LocalOpt = torch.optim.Adam(self.S.StructureParameters, lr=1e-2)
      self.LocalOpt_scheduler = ReduceLROnPlateau(
        optimizer=self.LocalOpt, factor=1 / 2.,
        mode='min', patience=self.temp_slope_opt_steps/25, threshold=0,
        cooldown=self.temp_slope_opt_steps/10, verbose=True, min_lr=1e-6)
    if self.load_structures_and_metrics != '':
      self.S.load_structures(self.load_structures_and_metrics)

    #############
    # main loop #
    #############
    print('Starting optimization')
    for step in tqdm(range(optimization_steps)):
      self.step = step
      if self.do_mtrain and step:
        self.SOpt_scheduler.step(self.mean_current_val)  # MTrain-Val
        # if self.LocalOpt is not None:
        #   self.LocalOpt_scheduler.step(self.mean_current_val) #MTrain-Val

      # with default values the midpoint @0.7%, end @0.005%
      ## temperature settings
      acc_rate = np.exp(self.initial_acc - 5.* step/self.temp_slope_opt_steps)
      if self.auto_temp and step:
        # Should I also divide by the number of features?
        print('Residual autocorrelation: ', self.residual_autocorr)
        if self.do_mtrain:
          temp = 2*np.mean(self.current_train)/(
              self.T.MTRAIN[0].TrainOutput.numel()
              *(1-self.residual_autocorr**2))
        else:
          temp = 2*np.mean(self.current_Mtrain)/(
              self.T.MTRAIN[0].TrainOutput.numel()
              *(1-self.residual_autocorr**2))
      elif self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
        temp *= temp_change
      else:
        temp = max(temp/temp_change, self.min_temp)

      #########################################
      # Simulated Annealing on Metatrain data #
      #########################################
      if self.encoder_input_type not in ['none', 'truth']:
        self.encoder.train()
      for module in self.L: module.train()
      self.BG_mode = 'train'
      self.initialize_mtrain_metrics_tracking()
      if self.do_mtrain:
        for i_batch, batch_datasets in enumerate(self.DL_Mtrain):
          #######################
          # Simulated Annealing #
          #######################
          # Update structure
          # for dataset in batch_datasets:
          #   self.S.update_structure(self.S.TrainStructures[dataset.structure_idx], step=self.step)
          batch_structures = [self.S.TrainStructures[dataset.structure_idx]
              for dataset in batch_datasets]
          updated_structures, train_loss, val_loss, train_loss_every, \
              val_loss_every, train_ans, val_ans, numcorr_SA_truth,\
              numcorr_enc_truth, numcorr_enc_SA, pred_probs, chosen, \
              accepted_structs, accepted_pred_probs, numcorr_bidir, MAML_g \
            = self.batched_bounce(batch_structures, batch_datasets, temp,
                do_grad=True)

          # update TrainStructures with new structures
          for structure, dataset in zip(updated_structures, batch_datasets):
            self.S.TrainStructures[dataset.structure_idx] = structure

          #### encoder step ####
          if (self.encoder_input_type not in ["none", "truth"] and
              accepted_pred_probs.shape[0] > 0):
            encoder_loss = self.encoder_loss(accepted_structs,
                accepted_pred_probs, len(batch_structures))
            self.encoder_step(encoder_loss, chosen)
          else:
            encoder_loss = torch.Tensor([0])

          # metrics tracking per batch
          self._update_mtrain_perf_tracking(train_loss, val_loss,
              train_loss_every, val_loss_every,
              encoder_loss.detach().cpu().numpy(),
              numcorr_SA_truth, numcorr_bidir, numcorr_enc_truth,
              numcorr_enc_SA)
          # self._update_mtrain_answers_tracking(batch_datasets, train_ans, val_ans)
          ####################
          # Gradient Descent #
          ####################
          if i_batch == 0:
            print(sum([p.norm(2.).item() for p in self.L.parameters()])**0.5)
          # torch.nn.utils.clip_grad_norm_(self.L.parameters(), 10.) #9.8 in exp
          self.SOpt.step()
          self.SOpt.zero_grad()
          if self.LocalOpt is not None:
            self.LocalOpt.step()
            self.LocalOpt.zero_grad()

      ##############################################
      # Simulated Annealing on MetaValidation data #
      ##############################################
      self.BG_mode = 'eval'
      if self.encoder_input_type not in ['none', 'truth']:
        self.encoder.eval()
      for module in self.L: module.eval()
      self._initialize_mval_metrics_tracking()
      for i_batch, batch_datasets in enumerate(self.DL_Mval):
        # Update structure
        # for dataset in batch_datasets:
        #   self.S.update_structure(
        #       self.S.ValStructures[dataset.structure_idx], step=self.step)
        batch_structures = [self.S.ValStructures[dataset.structure_idx]
            for dataset in batch_datasets]
        updated_structures, train_loss, val_loss, train_loss_every, \
            val_loss_every, train_ans, val_ans, numcorr_SA_truth,  \
            numcorr_enc_truth, numcorr_enc_SA, pred_probs, chosen, \
            accepted_structs, accepted_pred_probs, numcorr_bidir, MAML_g \
          = self.batched_bounce(
              batch_structures, batch_datasets, temp, do_grad=False)
        # update ValStructures with new structures
        for structure, dataset in zip(updated_structures, batch_datasets):
          self.S.ValStructures[dataset.structure_idx] = structure

        if self.plot_with_without_node and i_batch == 0:
          self.do_plot_with_without_node(batch_structures, batch_datasets)
        # encoder_loss
        if (self.encoder_input_type not in ["none", "truth"] and
            accepted_pred_probs.shape[0] > 0):
          encoder_loss = self.encoder_loss(
              accepted_structs, accepted_pred_probs,
              len(batch_structures)).detach().cpu().numpy()
        else:
          encoder_loss = np.array([0])

        # metrics tracking per batch
        self._update_mval_perf_tracking(train_loss, val_loss, train_loss_every,
            val_loss_every, encoder_loss, numcorr_SA_truth, numcorr_bidir,
            numcorr_enc_truth, numcorr_enc_SA)
        # self._update_mval_answers_tracking(batch_datasets, train_ans, val_ans)

      # 0-out optimizers (step in MTRAIN performed + dont want step from MVAL)
      self.SOpt.zero_grad()

      ###########################
      # stats tracking per step #
      ###########################

      # Simulated annealing
      self.writer.add_scalar('SA/temp', np.log10(temp).item(), self.step)
      self.writer.add_scalar('SA/acc_rate',
          np.log10(self.SA_running_acc_rate).item(), self.step)
      if self.S.find_node:
        # Visualize first 5 structures
        self.writer.add_scalar('mean_loss_find_node',
            self.S.find_node_dist/self.find_ct, self.step)
        for j in range(5):
          self.writer.add_scalar('mean_loss_find_node_'+str(j),
              np.mean(np.array(self.find_node_dist_rels[j])), self.step)
          self.writer.add_scalar('median_loss_find_node'+str(j),
              np.median(np.array(self.find_node_dist_rels[j])), self.step)

        colors = ['r', 'g', 'b', 'k', 'c']
        for i in range(100):
          dataset = self.T.MTRAIN[i]
          if not(i < 5 or np.sum(dataset.Edges[:4]) < 0.5):
            continue
          # print(i, np.sum(dataset.Edges[:4]))
          fig = plt.figure()
          for j in range(5):
            for k in range(dataset.TrainInput.shape[0]):
              plt.scatter(dataset.TrainInput[k,j,0].detach().cpu().numpy(),
                  dataset.TrainInput[k,j,1].detach().cpu().numpy(),
                  s=rcParams['lines.markersize']**2 * 4,
                  alpha=k/dataset.TrainInput.shape[0]/2.,
                  #np.linspace(0.,1.,dataset.TrainInput.shape[0]),
                  c=colors[j])
          pred = self.S.TrainStructures[
              dataset.structure_idx]['find_node_preds'][:,:1]
          for k in range(pred.shape[0]):
            plt.scatter(pred[k,0,0], pred[k,0,1], marker='*',
                s=rcParams['lines.markersize']**2 * 4,
                alpha=k/dataset.TrainInput.shape[0]/2.,
                # alpha=0.5, #np.linspace(0.,1.,dataset.TrainInput.shape[0]),
                c=colors[0])
          plt.xlim([-1,1])
          plt.ylim([-1,1])
          self.writer.add_figure('find_node_'+str(i), fig, self.step)
          plt.clf()
      self.update_metrics()
      # self.update_answers()
      if self.stop_edge_acc > 0:
        time_threshold = time.time() - 60.*self.stop_edge_acc
        for t in self.temporal_acc_SA_truth_mval.keys():
          if (t < time_threshold and self.temporal_acc_SA_truth_mval[t]+0.01 >
              self.current_acc_SA_truth_mval):
            early_kill = True
            break
        else: early_kill = False
      else: early_kill = False
      if (step % self.save_modules_every == 0 or early_kill
          or step == self.optimization_steps - 1):
        self.save_train_state()
      if early_kill: return



  def save_train_state(self):
    '''
    saves current modules to "models/{filename}"
    '''
    directory = "models"
    if not os.path.exists(directory):
      os.makedirs(directory)
    self.save_L(self.models_path)
    self.S.save_structures(self.models_path)

  ####################
  # Metrics tracking #
  ####################
  def initialize_mtrain_metrics_tracking(self):
    self.find_ct = 0.
    self.find_node_dist = 0.
    self.find_node_dist_rels = [[] for _ in range(5)]
    self.current_train = []
    self.current_val = []
    self.current_train_every = [[] for _ in self.everys]
    self.current_val_every = [[] for _ in self.everys]
    self.MTrain_norm_diff = []
    self.MTrain_cos_diff = []
    self.mtrain_numcorr_SA_truth = 0
    self.mtrain_bidir_correct = 0
    self.mtrain_encoder_losses = []
    self.mtrain_numcorr_enc_truth = 0
    self.mtrain_numcorr_enc_SA = 0

  def _initialize_mval_metrics_tracking(self):
    self.current_Mtrain = []
    self.current_Meval = []
    self.current_Mtrain_every = [[] for _ in self.everys]
    self.current_Meval_every = [[] for _ in self.everys]
    self.mval_numcorr_SA_truth = 0
    self.mval_bidir_correct = 0
    self.mval_encoder_losses = []
    self.mval_numcorr_enc_truth = 0
    self.mval_numcorr_enc_SA = 0

  def _update_mtrain_perf_tracking(self, train_loss, val_loss,
      train_loss_every, val_loss_every, encoder_loss, num_edges_correct,
      num_bi_corr, encoder_num_edges_correct, encoder_num_edges_proposal):
    # everys
    for i, (tle, vle) in enumerate(zip(train_loss_every, val_loss_every)):
      self.current_train_every[i].extend(tle)
      self.current_val_every[i].extend(vle)
    # encoder metrics
    self.mtrain_encoder_losses.append(encoder_loss)
    self.mtrain_numcorr_enc_truth += encoder_num_edges_correct
    self.mtrain_numcorr_enc_SA += encoder_num_edges_proposal
    # accuracy
    self.mtrain_numcorr_SA_truth += num_edges_correct
    self.mtrain_bidir_correct += num_bi_corr
    # loss
    self.current_train.extend(train_loss)
    self.current_val.extend(val_loss)

  def _update_mval_perf_tracking(self, train_loss, val_loss, train_loss_every,
      val_loss_every, encoder_loss, num_edges_correct, num_bi_corr,
      encoder_num_edges_correct, encoder_num_edges_proposal):
    # everys
    for i, (tle, vle) in enumerate(zip(train_loss_every, val_loss_every)):
      self.current_Mtrain_every[i].extend(tle)
      self.current_Meval_every[i].extend(vle)
    # encoder metrics
    self.mval_encoder_losses.append(encoder_loss)
    self.mval_numcorr_enc_truth += encoder_num_edges_correct
    self.mval_numcorr_enc_SA += encoder_num_edges_proposal
    # accuracy
    self.mval_numcorr_SA_truth += num_edges_correct
    self.mval_bidir_correct += num_bi_corr
    # loss
    self.current_Mtrain.extend(train_loss)
    self.current_Meval.extend(val_loss)

  def update_metrics(self):
    '''
    Updates several metrics after each step
    '''
    # accuracy: SA proposal / truth
    if self.do_mtrain:
      acc_SA_truth_mtrain = self.mtrain_numcorr_SA_truth / self.T.mtrain
      acc_SA_truth_mtrain = max(1 - acc_SA_truth_mtrain, acc_SA_truth_mtrain)
      self.writer.add_scalar('accuracy_SA_truth/Mtrain',
          acc_SA_truth_mtrain, self.step)
    acc_SA_truth_mval = self.mval_numcorr_SA_truth / self.T.mval
    acc_SA_truth_mval = max(1 - acc_SA_truth_mval, acc_SA_truth_mval)
    self.writer.add_scalar('accuracy_SA_truth/Mval',
        acc_SA_truth_mval, self.step)
    self.current_acc_SA_truth_mval = acc_SA_truth_mval
    self.temporal_acc_SA_truth_mval[time.time()] = acc_SA_truth_mval

    # accuracy: encoder proposal / SA proposal
    if self.do_mtrain:
      acc_enc_SA_mtrain = self.mtrain_numcorr_enc_SA / self.T.mtrain
      acc_enc_SA_mtrain = max(1 - acc_enc_SA_mtrain, acc_enc_SA_mtrain)
      if self.encoder_input_type == 'structure':
        self.writer.add_scalar('accuracy_encoder_SA/Mtrain',
            (acc_enc_SA_mtrain-0.8)/0.2, self.step)
      else:
        self.writer.add_scalar('accuracy_encoder_SA/Mtrain',
            acc_enc_SA_mtrain, self.step)
    acc_enc_SA_mval = self.mval_numcorr_enc_SA / self.T.mval
    acc_enc_SA_mval = max(1 - acc_enc_SA_mval, acc_enc_SA_mval)
    if self.encoder_input_type == 'structure':
      self.writer.add_scalar('accuracy_encoder_SA/Mval',
          (acc_enc_SA_mval-0.8)/0.2, self.step)
    else:
      self.writer.add_scalar('accuracy_encoder_SA/Mval',
          acc_enc_SA_mval, self.step)

    # accuracy: number bidirectional edges / total edges
    if self.do_mtrain:
      bidir_accuracy_mtrain = self.mtrain_bidir_correct / self.T.mtrain
      # bidir_accuracy_mtrain = max(1 - bidir_accuracy_mtrain, bidir_accuracy_mtrain) bidirectionality is well defined
      self.writer.add_scalar('accuracy_bidirectional/Mtrain',
          bidir_accuracy_mtrain, self.step)
    bidir_accuracy_mval = self.mval_bidir_correct / self.T.mval
    # bidir_accuracy_mval = max(1 - bidir_accuracy_mval, bidir_accuracy_mval) bidirectionality is well defined
    self.writer.add_scalar('accuracy_bidirectional/Mval', bidir_accuracy_mval, self.step)

    # accuracy: encoder proposal / truth
    if self.do_mtrain:
      acc_enc_truth_mtrain = self.mtrain_numcorr_enc_truth / self.T.mtrain
      acc_enc_truth_mtrain = max(1 - acc_enc_truth_mtrain, acc_enc_truth_mtrain)
      self.writer.add_scalar('accuracy_encoder_truth/Mtrain',
          acc_enc_truth_mtrain, self.step)
    acc_enc_truth_mval = self.mval_numcorr_enc_truth / self.T.mval
    acc_enc_truth_mval = max(1 - acc_enc_truth_mval, acc_enc_truth_mval)
    self.writer.add_scalar('accuracy_encoder_truth/Mval', acc_enc_truth_mval, self.step)

    # encoder loss
    if self.do_mtrain:
      self.writer.add_scalar('encoder_loss/Mtrain',
          np.mean(self.mtrain_encoder_losses), self.step)
    self.writer.add_scalar('encoder_loss/Mval',
        np.mean(self.mtrain_encoder_losses), self.step)

    # Differences between timesteps
    if self.do_mtrain and self.MTrain_norm_diff != []:
      self.writer.add_scalar('norm_diff/perc_10',
          np.percentile(self.MTrain_norm_diff, 10).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_90',
          np.percentile(self.MTrain_norm_diff, 90).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_10',
          np.percentile(self.MTrain_cos_diff, 10).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_90',
          np.percentile(self.MTrain_cos_diff, 90).item(), self.step)

    #Weight norms
    self.act_norms = [torch.norm(_) for m in self.L for _ in m.parameters()]
    self.norm_ratios = [(a/b).detach().cpu().numpy() for (a,b) in zip(
      self.act_norms, self.initial_norms)]
    self.writer.add_scalar('norm_ratios/mean',
        np.mean(self.norm_ratios).item(), self.step)
    self.writer.add_scalar('norm_ratios/std',
        np.std(self.norm_ratios).item(), self.step)

    #Error metrics
    if self.do_mtrain:
      self.mean_current_val = np.mean(self.current_val).item()
      self.writer.add_scalar('mean_loss/T_val', self.mean_current_val,
          self.step)
      self.writer.add_scalar('std_loss/T_val',
          np.std(self.current_val).item(), self.step)
      self.writer.add_scalar('min_loss/T_val',
          np.min(self.current_val).item(), self.step)
      if len(self.current_train):
        self.writer.add_scalar('mean_loss/T_train',
            np.mean(self.current_train).item(), self.step)
        self.writer.add_scalar('std_loss/T_train',
            np.std(self.current_train).item(), self.step)
        self.writer.add_scalar('min_loss/T_train',
            np.min(self.current_train).item(), self.step)
    if len(self.current_Mtrain):
      self.writer.add_scalar('mean_loss/V_train',
          np.mean(self.current_Mtrain).item(), self.step)
      self.writer.add_scalar('std_loss/V_train',
          np.std(self.current_Mtrain).item(), self.step)
      self.writer.add_scalar('min_loss/V_train',
          np.min(self.current_Mtrain).item(), self.step)
    if len(self.current_Meval):
      self.writer.add_scalar('mean_loss/V_val',
          np.mean(self.current_Meval).item(), self.step)
      self.writer.add_scalar('std_loss/V_val',
          np.std(self.current_Meval).item(), self.step)
      self.writer.add_scalar('min_loss/V_val',
          np.min(self.current_Meval).item(), self.step)

    # num_steps
    for i, e in enumerate(self.everys):
      # mean across all datasets
      if self.do_mtrain:
        self.writer.add_scalar('mean_loss_every_{}/T_val'.format(e),
            np.mean(self.current_val_every[i]).item(), self.step)
        self.writer.add_scalar('std_loss_every_{}/T_val'.format(e),
            np.std(self.current_val_every[i]).item(), self.step)
        self.writer.add_scalar('min_loss_every_{}/T_val'.format(e),
            np.min(self.current_val_every[i]).item(), self.step)
        if len(self.current_train_every):
          self.writer.add_scalar('mean_loss_every_{}/T_train'.format(e),
              np.mean(self.current_train_every[i]).item(), self.step)
          self.writer.add_scalar('std_loss_every_{}/T_train'.format(e),
              np.std(self.current_train_every[i]).item(), self.step)
          self.writer.add_scalar('min_loss_every_{}/T_train'.format(e),
              np.min(self.current_train_every[i]).item(), self.step)
      if len(self.current_Mtrain_every):
        self.writer.add_scalar('mean_loss_every_{}/V_train'.format(e),
            np.mean(self.current_Mtrain_every[i]).item(), self.step)
        self.writer.add_scalar('std_loss_every_{}/V_train'.format(e),
            np.std(self.current_Mtrain_every[i]).item(), self.step)
        self.writer.add_scalar('min_loss_every_{}/V_train'.format(e),
            np.min(self.current_Mtrain_every[i]).item(), self.step)
      if len(self.current_Meval_every):
        self.writer.add_scalar('mean_loss_every_{}/V_val'.format(e),
                               np.mean(self.current_Meval_every[i]).item(),
                               self.step)
        self.writer.add_scalar('std_loss_every_{}/V_val'.format(e),
                               np.std(self.current_Meval_every[i]).item(),
                               self.step)
        self.writer.add_scalar('min_loss_every_{}/V_val'.format(e),
                               np.min(self.current_Meval_every[i]).item(),
                               self.step)

  ####################
  # Answers tracking #
  ####################

  def initialize_answers(self):
    '''
    Create running answers  = search ensembles
    They weren't used in the paper, but may be in the future,
    as ensembles generally perform better than any single structure.
    :return:
    '''
    self.answers_running_score = 1e-5
    self.ans_eps = 1e-1  # 3e-2
    self.MTrainAnswers = [[None, None] for dataset in self.T.MTRAIN]
    self.MValAnswers = [[None, None] for dataset in self.T.MVAL]
    self.MTestAnswers = [[None, None] for dataset in self.T.MTEST]
    self.OldMTrainAnswers = [[None, None] for dataset in self.T.MTRAIN]

  def _update_mtrain_answers(self, batch_datasets, train_ans, val_ans):
    for i, dataset in enumerate(batch_datasets):
      idx = dataset.structure_idx
      if self.MTrainAnswers[idx][0] is None:
        self.MTrainAnswers[idx][0] = train_ans[i] * self.ans_eps
      else:
        self.MTrainAnswers[idx][0] = ((1 - self.ans_eps) * self.MTrainAnswers[idx][0] + train_ans[i] * self.ans_eps)
      if self.MTrainAnswers[idx][1] is None:
        self.MTrainAnswers[idx][1] = val_ans[i] * self.ans_eps
      else:
        self.MTrainAnswers[idx][1] = ((1 - self.ans_eps) * self.MTrainAnswers[idx][1] + val_ans[i] * self.ans_eps)
    # See differences
    if self.OldMTrainAnswers[idx][0] is not None:
      self.MTrain_norm_diff.append(
        np.mean(np.linalg.norm(self.OldMTrainAnswers[idx][0] - train_ans[i], axis=1)))
      self.MTrain_cos_diff.append(
        torch.mean(self.CosDist(torch.FloatTensor(
          torch.from_numpy(self.OldMTrainAnswers[idx][0]) - dataset.TrainOutput.cpu()),
          torch.FloatTensor(
            torch.from_numpy(train_ans[i]) - dataset.TrainOutput.cpu()))).data.cpu().numpy())
    self.OldMTrainAnswers[idx][0] = train_ans[i]

  def _update_mval_answers(self, batch_datasets, train_ans, val_ans):

    for i, dataset in enumerate(batch_datasets):
      idx = dataset.structure_idx
      if self.MValAnswers[idx][0] is None:
        self.MValAnswers[idx][0] = train_ans[i] * self.ans_eps
      else:
        self.MValAnswers[idx][0] = ((1 - self.ans_eps) * self.MValAnswers[idx][0] + train_ans[i] * self.ans_eps)
      if self.MValAnswers[idx][1] is None:
        self.MValAnswers[idx][1] = val_ans[i] * self.ans_eps
      else:
        self.MValAnswers[idx][1] = ((1 - self.ans_eps) * self.MValAnswers[idx][1] + val_ans[i] * self.ans_eps)

  def update_answers(self):
    self.answers_running_score = (self.answers_running_score * (1. - self.ans_eps) + self.ans_eps)
    ensemble_train = np.mean([np.mean(
      (self.MTrainAnswers[i][0] / self.answers_running_score
       - self.T.MTRAIN[i].TrainOutput.cpu().numpy()) ** 2)
                              for i in range(self.T.mtrain)])
    ensemble_val = np.mean([np.mean(
      (self.MTrainAnswers[i][1] / self.answers_running_score
       - self.T.MTRAIN[i].ValOutput.cpu().numpy()) ** 2)
                            for i in range(self.T.mtrain)])
    ensemble_Mtrain = np.mean([np.mean(
      (self.MValAnswers[i][0] / self.answers_running_score
       - self.T.MVAL[i].TrainOutput.cpu().numpy()) ** 2)
                               for i in range(self.T.mval)])
    ensemble_Mval = np.mean([np.mean(
      (self.MValAnswers[i][1] / self.answers_running_score
       - self.T.MVAL[i].ValOutput.cpu().numpy()) ** 2)
                             for i in range(self.T.mval)])
    self.writer.add_scalar('ensemble/train', ensemble_train.item(), self.step)
    self.writer.add_scalar('ensemble/val', ensemble_val.item(), self.step)
    self.writer.add_scalar('ensemble/Mtrain', ensemble_Mtrain.item(), self.step)
    self.writer.add_scalar('ensemble/Mval', ensemble_Mval.item(), self.step)

  ###########################################
  # Experiment restarting utility functions #
  ###########################################

  def save_L(self, directory=None):
    '''
    Saves ModuleList
    '''
    if directory is None: directory = 'moduleList'
    if not os.path.exists(directory):
      os.makedirs(directory)
    for i_m, module in enumerate(self.L):
      torch.save(module.state_dict(), os.path.join(directory , str(i_m)))

    if self.encoder_input_type not in ['none', 'truth']:
      torch.save(self.encoder.state_dict(), os.path.join(directory, 'encoder'))


  def load_L(self, directory=None):
    '''
    Loads ModuleList
    '''
    if directory is None: directory = 'moduleList-'
    for i_m, module in enumerate(self.L):
      self.L[i_m].load_state_dict(torch.load(
        os.path.join(directory, str(i_m))))
    if self.encoder_input_type not in ['none', 'truth']:
      self.encoder.load_state_dict(torch.load(
        os.path.join(directory, 'encoder')))
