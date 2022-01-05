DataGen_HIV <- function(Gparams, maxsamples, big_max, depth_wise_sampling = FALSE, save_prefix = 'Training_Data/') {
  # depth_wise_sampling here for API compatability
  library(abind)
  library(R.matlab)
  library(tictoc)
  library(parallel)
  library(iterators)
  library(foreach)

  RO <- Gparams$values_RO()
  ts <- Gparams$values_timestep()
  lambda <- Gparams$logspace_lambda()
  nbins <- Gparams$set_RO_num() * Gparams$set_ts_num()
  itermax <- floor(maxsamples / (nbins)) #get bin sizes
  max_size <- itermax * (nbins) # get the number of samples to generate, is divisible by itermax, theta_n, and mu_n
  filenames <- list() # empty list
  bins <- seq(1, (nbins))

  dim(bins) <- c(Gparams$RO_num, Gparams$ts_num)  # force into correct shape
  print(bins)
  print(dim(bins))

  print('reach for loop')

  for (big_counter in 1:big_max) {
    #Create some objects to store data within
    labels <- matrix(0, max_size, 1) #For one-hot encodings of bins
    data <- array(0, dim = c(max_size, Gparams$NSamples, Gparams$NSamples)) #store the data
    values <- matrix(0, max_size, 2) # save the parameter values (R0 and ts)
    indexes <- matrix(0, max_size, 2) # save the Counter indices (i.e. the grid coordinates)
    NPop <- matrix(0, max_size, 1) # save the population size data
    tic() #outer tic
    for (tsCounter in 1:Gparams$ts_num) {
      for (ROCounter in 1:Gparams$RO_num) {
        print(c(ROCounter, tsCounter))
        tic() #inner tic
        inds <- seq((bins[ROCounter, tsCounter] - 1) * itermax + 1, (bins[ROCounter, tsCounter]) * itermax)
        #This calls a parallelized loop by default
          tmp_data <- bigloop_HIV(NSamples = Gparams$NSamples,
                                  R0 = RO[ROCounter],
                                  totalStep = ts[tsCounter],
                                  itermax = itermax)

        data[inds, ,] <- abind(tmp_data$Matrices, along = 0)
        NPop[inds] <- tmp_data$NPop
        indexes[inds, 1] <- ROCounter
        indexes[inds, 2] <- tsCounter

        values[inds, 1] <- tmp_data$R0
        values[inds, 2] <- ts[tsCounter]
        labels[inds] <- bins[ROCounter, tsCounter]
        print(bins[ROCounter, tsCounter]) #display me!
        toc(log = TRUE)
      }
    }

    toc(log = TRUE)

    print(
      paste(
        "Reach save number ",
        toString(big_counter),
        ", the simulation has finished of",
        toString(big_max),
        "iterations total.",
        sep = " "
      )
    )
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    settings_list <-
      paste(
        "Generated from Kupperman et al.'s HIV model,",
        'NSamples' = c(Gparams$NSamples),
        'max_size' = c(max_size)
      )
    settings_log <-
      paste0(
        'Sample generated at time: ',
        toString(lgd),
        ', NSamples = ',
        toString(Gparams$NSamples),
        ', max_size = ',
        toString(max_size),
        'used a non-sequential model'
      )

    filenameMat <- paste0(save_prefix, 'TrainingSet', lgd, '-dsize-', toString(Gparams$NSamples), '.mat')  # Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = data,
      labels = labels,
      values = values,
      indexes = indexes,
      settings_list = settings_list,
      log = settings_log,
      lambda = lambda,
      NPop = NPop
    )
    # access log with writeLines(log)
    print(
      paste(
        "Loop iteration",
        toString(big_counter),
        "has finished of",
        toString(big_max),
        "iterations total.",
        sep = " "
      )
    )
    filenames <- append(filenames, filenameMat)
  }
  Gparams$filenames <- append(Gparams$filenames, filenames)

}
