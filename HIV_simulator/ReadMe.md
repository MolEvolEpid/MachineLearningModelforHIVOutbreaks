# HIV Simulation in R

This directory contains all R codes. This directory contains

1. `HIVsim.R` The simulator codes. Multiple options exist for specifying the behavior. Options for single and 
   multi-threaded parallelized simulations exist.
    
2. `DataGen_HIV.R` Dispatcher and controller for handling parameter sweeps, randomization, and concatenation.

3. `Data_Generator_class.R` Class for storing parameter values. Contains utility methods. 

4. `RunMe_15.R` Top-level R-script for data generation. 

The Matlab file format prevents storing any object larger than 2Gb. To avoid restrictions on the size of a large 
training set, we generate multiple smaller training files. Additionally, we attempt to avoid doing both parallel 
simulations and storing large matrices in memory at the same time on the same job, a feature we have found practical 
when running multiple jobs on the same machine.

Data outputs should be serialized into a Matlab `.mat` data file using the `R.matlab` library.

Data output contains 8 user defined fields. Not all fields are saved when imported to Python. We give the `key` 
attached to each field and describe the contents.

* `matrices` (array) The 3d tensor of matrices with shape `(n,k,k)`. Stores the pairwise distance matrices we will 
  use to train the model, `n` is the number of matrices sampled, `k` is 
  number of samples used to generate each pairwise distance matrix.
* `NPop` (vector) Store the final population size actually attained. 
* `labels` (vector of `n` elements) Precomputed labels (1-based integers) from the striding pattern of binned 
  R0-nought and time stepping. 
  Disregard when $R_0$ is randomized within the simulation.  
* `indexes` (matrix with shape `(n,k)`) Store your own 1-based indices for parameters. Use these to rebuild labels 
  in Python so you can vary 
  other parameters. We disregard the first column (storing R-nought) in our python code.
* `values` (matrix with shape `(n,k)`) Values from the simulation used, corresponding to the same rows/columns of 
  `indexes`, but store 
  values 
  returned from the simulation.
* `settings_list` (string) Store metadata from simulation that may not be stored in values.
* `log` (string) Log file manipulation events (join, randomize, sort) and when they occur.  
