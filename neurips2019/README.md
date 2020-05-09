# Neural Relational Inference with Fast Modular Meta-learning
You can find the data [here](http://lis.csail.mit.edu/alet/neurips2019_data/); put it inside a folder called ```data```.

The current code is not compatible with the original compositions. The main differences are:
* This code supports batching across datasets; for a non-GNN composition the author has to provide a way to batch multiple structures evaluated in multiple datasets.
* As before, the algorithm has a proposal function to change the structures. However, now this function can be learned.

We have checked that results on this code are the same (actually, a bit better) than those published at NeurIPS, albeit the optimization is slower than with Pytorch 1.0.

To reproduce them yourself for the full 50k datasets (takes about 1 day for springs,7 for charged) run ```run_springs.sh``` and ```run_charged.sh``` respectively
![alt text](https://github.com/FerranAlet/modular-metalearning/blob/master/neurips2019/images/summary_charged.png "Summary charged experiments")
![alt text](https://github.com/FerranAlet/modular-metalearning/blob/master/neurips2019/images/summary_springs.png "Summary charged springs")
#### Comments on curves:
* For the charged experiments losses are below 1e-3 and 3e-3 and accuracy is above 89%; slightly better than the paper results; probably a result of training longer.
* Note that the accuracy of the proposal function (without simulated annealing) is very similar to the results of Kipf et al., which makes sense given it plays the role of their edge prediction function.
* The encoder ran into an instability for springs, by that time the loss and accuracies were already better than the published results. We conjecture that since its accuracy was above 99% the proposal function learned to be very confident; with an unlucky accept of a bad proposal during Simulated Annealing incurred into a very big loss and thus very bad update. *This did not affect the overall performance* since Simulated Annealing just rejected the bad proposals.
