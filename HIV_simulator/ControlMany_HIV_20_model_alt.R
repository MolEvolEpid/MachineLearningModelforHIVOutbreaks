#ControlMany_HIV
#This is a data generation script that creates arrays of pairwise_differance matrices
#START FROM THIS SCRIPT
rm(list = ls())#clear environment

#setwd("~/Documents/Projects/ViralPhylo/ViralP")
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
#source(here("DataGen_NS.R"))
source(here("HIVsim.R"))
#source(here("RapidSim.R"))
source(here("Data_Generator_class.R"))
for(val in 1:5){
  Gparams <- Data_Generator(
    
    #detectable mutation rate generation
    lambda_large = -1,
    lambda_small = -4,
    lambda_number = 1,#DONT SET TO 3
    #HIV unique simulation paramers
    use_linear_timesteps = FALSE,
    ts_list_in = c(0,2,10),
    ts_min = 0,
    ts_max = 10,
    ts_num = 2,
    
    use_linear_RO = TRUE,
    RO_list_in = c(0,0),
    RO_min = 5,
    RO_max = 15,
    RO_num = 1,
    
    #Generic simulation parameters 
    N = 500,   # Set only for API intercompatability
    NSamples = 20,
    preseed_mutants = TRUE,
    
    constant_s_step = TRUE,
    filenames = list()
  )#END Gparams
  
  # Define simulation split/length --------
  maxsamples <- 66668
  tt_split <- .1
  if ((tt_split < 0 ||
       tt_split > 1))
    stop("test/train split (tt_split) must be between 0 and 1 inclusive")
  nfolds_test <- 2
  nfolds_test <- as.integer(nfolds_test)
  if (as.integer(nfolds_test) <= 0)
    stop("nfolds must be at least one")
  nfolds_train <- 2
  nfolds_train <- as.integer(nfolds_train)
  
  Gparams$set_s_mod_number()#$set s_mod_number
  #print(Gparams$s_mod_number)
  
  Hparams <- Gparams$copy()#duplicate the object
  test_samples <-30000 / nfolds_test
    # ceiling(tt_split * maxsamples / nfolds_test) #absolute maximum number of data to generate
  train_samples <- 60000 / nfolds_train
    #ceiling((1 - tt_split) * maxsamples / nfolds_train) #absolute maximum number of data to generate
  itermax_test <- nfolds_test#how many iterations to do
  itermax_train <- nfolds_train
  
  if (val == 1) {
    if (tt_split > 0) {
    DataGen_HIV(Gparams, test_samples, itermax_test, model='alt')
  }
  #We still have Gparams, all of our new file names are stored in Gparams$filenames
    shuffle_data(Gparams$join_datasets(), prefix='', ofix='W20Training_Data/test-')
    #we already have the file extension prefix saved
  }
  #Now generate the test set
  DataGen_HIV(Hparams, train_samples, itermax_train, model='alt')
  
  shuffle_data(Hparams$join_datasets(), prefix = '', ofix='W20Training_Data/train-')
  #we already have the file extension prefix saved
}
