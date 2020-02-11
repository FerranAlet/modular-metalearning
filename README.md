# Modular meta-learning

Code for the papers [Modular meta-learning](https://arxiv.org/abs/1806.10166) and [NRI with Fast Modular meta-learning](https://papers.nips.cc/paper/9353-neural-relational-inference-with-fast-modular-meta-learning) (in the [neurips_2019 folder](https://github.com/FerranAlet/modular-metalearning/tree/master/neurips2019))

The code has been refactored to make it easy to extend with your own variations.

## Prerequisites
* [PyTorch](https://pytorch.org/get-started/locally/)
* [tensorboardX](https://github.com/lanpa/tensorboardX) the equivalent to tensorboard for pytorch
* (Optional) CUDA GPU support. If you don't have it, you can comment all lines with nn_device and cuda() references.

## I want to create my own composition; what should I do?
Subclass both [Composer](https://github.com/FerranAlet/modular-metalearning/blob/master/composition.py) and [Structure](https://github.com/FerranAlet/modular-metalearning/blob/master/structure.py).
* **Composer**: defines how to compute the output from the input and a collection of modules.
* **Structure**: defines a structure initialization and a structure proposal function.

The simplest example of how to do this is the [sum](https://github.com/FerranAlet/modular-metalearning/blob/master/sum_composer.py). We will shortly add a more complex Graph Neural Network composer.

Finally, you should import your new file in [modular_main](https://github.com/FerranAlet/modular-metalearning/blob/master/modular_main.py) and add it to the possible compositions at towards end of the function.


## Example runs
First you have to generate the datasets using  [create_functions_datasets](https://github.com/FerranAlet/modular-metalearning/blob/master/create_functions_datasets.py). For instance to create the functions dataset:
```
python create_functions_datasets.py --mode sum --not_alone --out_file sums.hdf5
```
To create the sines dataset:
```
python create_functions_datasets.py --mode sines --out_file sines.hdf5
```
The file has a couple flags to customize how many elements per dataset and how many datasets to create.

Then you run [modular_main](https://github.com/FerranAlet/modular-metalearning/blob/master/modular_main.py), which relies on the class defined in the [modular_metalearning](https://github.com/FerranAlet/modular-metalearning/blob/master/modular_metalearning.py) file.
For example to run BounceGrad in the sums dataset:
```
python modular_main.py --type_modules affine-1-16-16-1,affine-1-16-1 --num_modules 10,10 --composer sum-2 --meta_lr 0.003 --plot_name BounceGrad --limit_data 80 --optimization_steps 3000 --split_by_file --meta_split 90,10,0 --data_split 20,80,0 --data_desc HDF5@sums.hdf5 --meta_batch_size 32
```
or to run MAML in the sines dataset:
```
python modular_main.py --type_modules affine-1-64-64-1 --num_modules 1 --composer functionComposition-1 --meta_lr 0.001 --plot_name MAML --limit_data 80 --optimization_steps 5000 --split_by_file --meta_split 90,10,0 --data_split 20,80,0 --data_desc HDF5@sines.hdf5 --meta_batch_size 16 --max_datasets 300 --MAML --MAML_step_size 0.1
```
## Algorithm options
You can run either of the 4 algorithms described in the paper:
* **Pooled**: monolithic structure with fixed weights.
* **MAML**: monolithic structure with adaptable weights.
* **BounceGrad**: adaptable modular structructure with fixed weights.
* **MoMA**: adaptable modular structure with adaptable weights.

The 4 algorithms come from two independent decisions: whether to have more than one module and whether to add MAML.

All options are flags in [modular_main](https://github.com/FerranAlet/modular-metalearning/blob/master/modular_main.py). To activate MAML use the flag ```--MAML``` and you also have access to customizing the step size and number of steps.

To not have modularity use ```--composer functionComposition-1```, since a [functionComposition](https://github.com/FerranAlet/modular-metalearning/blob/master/functioncomposition_composer.py) of depth 1 is just applying that one function.

To use modularity simply specify your composer and add more than one module in the ``--num_modules`` flag. If your composition requires more than one module type describe both the types and the numbers separated by commas, for example:
```--type_modules relu-1-64-64-1,affine-1-16-1 --num_modules 5,10```.

## How many modules and how many optimization steps?
In our experience the number of modules is a pretty flexible hyperparameter; we recommend putting it 10-30% above the number of modules you think would be needed in an optimal split. For example, in the functions experiments there are 16 functions, thus needing 16 modules; we rounded it up to 20 modules. Experiments with little less and little more did not affect much.

Beware that adding modules makes the use of each module less frequent, and therefore you may need to adjust ``--meta_lr`` and ``--meta_batch_size`` accordingly.

## Some works using this code
* [Learning Quickly to Plan Quickly Using Modular Meta-Learning](https://arxiv.org/abs/1809.07878) Chitnis et al.
* [Modular meta-learning in Abstract Graph Networks for combinatorial generalization](https://arxiv.org/pdf/1812.07768.pdf); Alet et al. ; NeurIPS meta-learning workshop 2018

## If you find this code useful in your research, please consider citing
```
@inproceedings{alet2018modular,
    author={Ferran Alet, Tomas Lozano-Perez, Leslie Pack Kaelbling},
    title={Modular meta-learning},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2018},
    url={http://lis.csail.mit.edu/pubs/alet-corl18.pdf}
}
@incollection{NIPS2019_9353,
title = {Neural Relational Inference with Fast Modular Meta-learning},
author = {Alet, Ferran and Weng, Erica and Lozano-P\'{e}rez, Tom\'{a}s and Kaelbling, Leslie Pack},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {11804--11815},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/9353-neural-relational-inference-with-fast-modular-meta-learning.pdf}
}
```


## Questions? Suggestions?
Please email me at alet(at)mit(dot)edu.
