#ControlMany_HIV
# This is a data generation script
# that creates arrays of pairwise_differance matrices

# Define section -------------
library(here)
library(abind)
library(R.matlab)
library(tictoc)
library(parallel)
library(iterators)
library(foreach)
library(seriation)

source(here("Utility_functions.R"))
source(here("DataGen_HIV.R"))
source(here("HIVsim.R"))
source(here("Data_Generator_class.R"))

for(val in 1:5){ # number of data sets
  Gparams <- Data_Generator(
    NSamples = 15,
    #HIV unique simulation paramers
    use_linear_timesteps = FALSE,
    ts_list_in = c(0,2,10),
    ts_min = 0,
    ts_max = 10,
    ts_num = 2,

    filenames = list(),

    # The following are set for API intercompatability with another project

    # detectable mutation rate generation
    lambda_large = -1,
    lambda_small = -4,
    lambda_number = 1,


    use_linear_RO = TRUE,
    RO_list_in = c(0, 0),
    RO_min = 5,
    RO_max = 15,
    RO_num = 1,
    # Generic simulation parameters
    N = 500,   # Set only for API intercompatability
    preseed_mutants = TRUE,
    constant_s_step = TRUE
  )#END Gparams

  # Define simulation split/length --------
  nfolds_test <- 2
  nfolds_test <- as.integer(nfolds_test)
  if (as.integer(nfolds_test) <= 0)
    stop("nfolds must be at least one")
  nfolds_train <- 2
  nfolds_train <- as.integer(nfolds_train)
  Gparams$set_s_mod_number()#$set s_mod_number - for API intercompatability  
  Hparams <- Gparams$copy()#duplicate the object
  # How many samples per fold to generate
  test_samples <- 30000 / nfolds_test
  train_samples <- 60000 / nfolds_train
  # Update variable names for interation scheme
  itermax_test <- nfolds_test#how many iterations to do
  itermax_train <- nfolds_train

  # Only run validation data generation once
  if (val == 1) {
    if (tt_split > 0) {
    DataGen_HIV(Gparams, test_samples, itermax_test)
  }
  # Shuffle the data and save it to storage
  shuffle_data(Gparams$join_datasets(), prefix='',
  ofix='../Example_Data/Exponential_exits_mu67_synth/Generated_data_15/TEST/W15-TEST-')
    #we already have the file extension prefix saved
  }
  #Now generate the test set
  DataGen_HIV(Hparams, train_samples, itermax_train)
  shuffle_data(Hparams$join_datasets(), prefix = '',
  ofix='../Example_Data/Exponential_exits_mu67_synth/Generated_data_15/TRAIN/W15-TRAIN-')
  #we already have the file extension prefix saved
}
# Now clean the temp files we made along the way.
do.call(file.remove, list(list.files("./Training_Data/", full.names = TRUE)))
