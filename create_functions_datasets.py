import random
import h5py
import matplotlib.pyplot as plt
import argparse
import numpy as np

OPS = {
    'abs': lambda x,y : np.abs(x),
    'arcsinh': lambda x,y : np.arcsinh(y*x)/np.arcsinh(np.abs(y)), #y = 10
    'arctan': lambda x,y : np.arctan(y*x)/np.arctan(np.abs(y)), #y = 10
    'cbrt': lambda x,y: np.cbrt(x),
    'ceil': lambda x,y: np.ceil(x),
    'cos': lambda x,y : np.cos(y*np.pi*x), #y = 3
    'cosh': (lambda x,y : np.cosh(y*x)/np.cosh(np.abs(y))), #y = 4
    'exp2': (lambda x,y : np.exp2(y*x)/np.exp2(np.abs(y))), #y = 5
    'floor': lambda x,y: np.floor(x),
    'rint': lambda x,y: np.rint(x),
    'sign': lambda x,y: np.sign(x),
    'sin': (lambda x,y : np.sin(y*np.pi*x)), #y = 3
    'sinc': (lambda x,y : np.sinc(y*np.pi*x)), #y = 3
    'square': lambda x,y: np.square(x),
    'tanh': lambda x,y: np.tanh(x),
    'id': (lambda x,y : x),
    }

def special_sine(x,y,z): return np.sin(np.pi*(y*x+z))

LIST_OPS = sorted([o for o in OPS])
inline_ops = ['*', '+']

def main(args):
  def maybe_add_plot(ax_counter, x,y):
    if random.random() < 0.025 and ax_counter<9:
      ax[ax_counter//3][ax_counter%3].plot(x,y)
      ax[ax_counter//3][ax_counter%3].yaxis.set_ticks([-1,0,1])
      ax[ax_counter//3][ax_counter%3].xaxis.set_ticks([-1,0,1])
      ax[ax_counter//3][ax_counter%3].set_ylim((-1.1,1.1))
      return ax_counter + 1
    else: return ax_counter

  def run_sines_alet_etal(args):
    # As described in https://arxiv.org/abs/1806.10166
    # Varies frequency and phase
    ax_counter = 0
    for dataset in range(args.meta_datasets):
      freq = np.random.uniform(low = 0.1, high = 5.0)
      phase = np.random.uniform(low=0., high = np.pi)
      y = special_sine(x, freq, phase)
      ori_name = (str(freq).replace('.','d') + '_' +
          str(phase).replace('.','d'))
      DATA[ori_name+'-IN'] = x
      DATA[ori_name+'-OUT'] = y
      ax_counter = maybe_add_plot(ax_counter, x,y)

  def run_sines_finn_etal(args):
    # As described in https://arxiv.org/abs/1703.03400
    # Varies amplitude and phase
    # Not part of original experiments in https://arxiv.org/abs/1806.10166
    # because of a misunderstanding on our part.
    ax_counter = 0
    x = np.reshape(np.arange(-5., 5., 2./args.limit_data), (-1,1))
    for dataset in range(args.meta_datasets):
      amplitude = np.random.uniform(low = 0.1, high = 5.0)
      phase = np.random.uniform(low=0., high = np.pi)
      y = amplitude * special_sine(x, 1., phase)
      ori_name = (str(amplitude).replace('.','d') + '_' +
          str(phase).replace('.','d'))
      DATA[ori_name+'-IN'] = x
      DATA[ori_name+'-OUT'] = y
      ax_counter = maybe_add_plot(ax_counter, x,y)

  def run_functions(args):
    ax_counter = 0
    c = 4.
    for a in LIST_OPS:
      for b in LIST_OPS:
        if b < a: continue #sum is commutative
        if args.not_alone and b==a: continue
        print(a,b)
        ori_name = a + '_' + b
        y = (OPS[a](x, c*o) + OPS[b](x,c*o))/2.
        DATA[ori_name+'-IN'] = x
        DATA[ori_name+'-OUT'] = y
        ax_counter = maybe_add_plot(ax_counter, x,y)

  DATA = {}
  x = np.reshape(np.arange(-1., 1., 2./args.limit_data), (-1,1))
  o = np.ones_like(x)
  fig, ax = plt.subplots(nrows = 3, ncols=3)

  if args.mode == 'sines': run_sines_alet_etal(args)
  elif args.mode == 'sines-finn': run_sines_finn_etal(args)
  elif args.mode == 'sum': run_functions(args)
  else: raise NotImplementedError

  print(len(DATA))
  plt.show()

  # Plots 2 images visualizing a few sumed functions.
  if args.plot_metalearning_slide:
    #Plot 4 meta-training
    funs = [['abs','cos'], ['sign','square'],
        ['id','exp2'], ['id','cos']]
    fig, ax = plt.subplots(nrows=1, ncols=4)
    for i in range(4):
      x = np.random.uniform(-1,1, 8)
      y = np.ones_like(x)*4.
      ax[i].scatter(x, (OPS[funs[i][0]](x,y)+OPS[funs[i][1]](x,y))/2.)
      ax[i].set_ylim([-1.2,1.2])
      ax[i].set_xlim([-1,1])
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      ax[i].set_aspect(1.)
    plt.savefig('meta-train.png')
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    for i in range(4):
      x = np.random.uniform(-1,1,100)
      y = np.ones_like(x)*4.
      ax[i].scatter(x, (OPS[funs[i][0]](x,y)+OPS[funs[i][1]](x,y))/2.)
      ax[i].set_ylim([-1.2,1.2])
      ax[i].set_xlim([-1,1])
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      ax[i].set_aspect(1.)
    plt.savefig('meta-test.png')

  # Saves to file
  h = h5py.File(args.out_file)
  for k,v in DATA.items():
    h.create_dataset(k, data=v)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--plot_metalearning_slide',
      dest='plot_metalearning_slide', action='store_true',
      help='whether to make plots for a meta-learning visualization')
  parser.add_argument('--not_alone', dest='not_alone', action='store_true',
      help='whether a function can appear alone')
  parser.add_argument('--parameter', dest='parameter', action='store_true',
      help='whether to change constants to several parameter options')
  parser.add_argument('--mode', dest='mode',
      default = 'sum', help='type of function; in [sines, sines-finn,sum]')
  parser.add_argument('--limit_data', dest='limit_data', type=int,
      default = 1000, help='maximum number of elements per dataset')
  parser.add_argument('--meta_datasets', dest='meta_datasets', type=int,
      default = 1000, help='number of metadatasets for sines datasets')
  parser.add_argument('--out_file', dest='out_file', default='out_file.hdf5',
      help='directory with bvh files')
  args = parser.parse_args()
  main(args)

