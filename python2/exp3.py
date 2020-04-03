'''
Author: Eric Aislan Antonelo
email: eric.antonelo@gmail.com
- Part of the Minicourse on Echo State Networks
Last updated on December 2017
'''

import mdp
import Oger
import scipy as sp

''' Example of doing a grid-search
Runs the NARMA 30 task for input scaling values = X to Y with Z stepsize and spectral radius = X to Y with stepsize Z
'''
inputs, outputs = Oger.datasets.narma30()

data = [[], zip(inputs, outputs)]

# construct individual nodes
reservoir = Oger.nodes.ReservoirNode(output_dim=100)
readout = Oger.nodes.RidgeRegressionNode()

# build network with MDP framework
flow = mdp.Flow([reservoir, readout])


# Nested dictionary
gridsearch_parameters = {reservoir:{'input_scaling': [], 'spectral_radius': [], '_instance':range(5)}}


# Instantiate an optimizer
opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)

# Do the grid search
#opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5)
opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.leave_one_out)


opt.plot_results([(reservoir, '_instance')])  # OBS: has to hack Oger toolbox, for this to work
# Get the optimal flow and run cross-validation with it 
opt_flow = opt.get_optimal_flow(verbose=True)


print 'Performing cross-validation with the optimal flow. Note that this error can be slightly different from the one reported above due to another division of the dataset. It should be more or less the same though.'

errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.leave_one_out)
print 'Mean error over folds: ' + str(sp.mean(errors))


