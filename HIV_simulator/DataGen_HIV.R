DataGen_HIV <- function(Gparams, maxsamples, big_max, depth_wise_sampling=FALSE, save_prefix='Training_Data/') {
# depth_wise_sampling samples into a single simulation, rather than across simulations
# default behavior is to sample accross simulations
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
  #print("Flag1!")
  #print(Gparams$s_mod_number)
  #if (Gparams$include_s_zero)
  #{
  #  nbins <- Gparams$s_mod_number * Gparams$mu_number + 1
  #} else {
  #  nbins <- Gparams$s_mod_number * Gparams$mu_number
  #}
  itermax <-
    floor(maxsamples / (nbins))#get bin sizes
  max_size <-
    itermax * (nbins)#get the number of samples to generate, is divisible by itermax, theta_n, and mu_n
  #if (Gparams$include_s_zero)
  #{
  #  max_size <- max_size + 1
  #  #increment the number of bins/labels
  #}
  
  filenames <- list()#empty list
  #so we can create our labels
  bins <- seq(1, (nbins))
  #if (Gparams$include_s_zero) {
  #  bins <- bins + 1
  #}
  
  dim(bins) <-
    c(Gparams$RO_num, Gparams$ts_num)#force into correct shape
  print(bins)
  print(dim(bins))
  
  print('reach for loop')
  
  for (big_counter in 1:big_max) {
    #Create some objects to store data within
    labels <- matrix(0, max_size, 1) #For one-hot encodings of bins
    data <-
      array(0, dim = c(max_size, Gparams$NSamples, Gparams$NSamples)) #store the data
    values <-
      matrix(0, max_size , 2) #save the parameter values (R0 and ts)
    indexes <-
      matrix(0, max_size, 2) #save the Counter indices (i.e. the grid coordinates)
    NPop <-
      matrix(0, max_size, 1) #save the population size data
    tic() #outer tic
    #Just do this bloc once
    #for (lcounter in 1:Gparams$lambda_number) {
      for (tsCounter in 1:Gparams$ts_num) {
        for (ROCounter in 1:Gparams$RO_num) {
          print(c(ROCounter, tsCounter))
          tic()#inner tic
          inds <-
            seq((bins[ROCounter, tsCounter] - 1) * itermax + 1, (bins[ROCounter, tsCounter]) *
                  itermax)
          #This calls a PARALLELIZED LOOP
          if(depth_wise_sampling){
          tmp_data <- bigloop_HIV_depthwise(
                NSamples = Gparams$NSamples,
                R0 = RO[ROCounter],
                totalStep = ts[tsCounter],
                itermax = itermax)
          }else{
          tmp_data <-
              bigloop_HIV(
                NSamples = Gparams$NSamples,
                R0 = RO[ROCounter],
                totalStep = ts[tsCounter],
                itermax = itermax) #,
            #  along = 0
            #) 
          }
          data[inds, , ] <- abind(tmp_data$Matrices, along=0)
          NPop[inds] <- tmp_data$NPop
          indexes[inds, 1] <- ROCounter
          indexes[inds, 2] <- tsCounter

          values[inds, 1] <- RO[ROCounter]
          values[inds, 2] <- ts[tsCounter]
          labels[inds] <- bins[ROCounter, tsCounter]
          print(bins[ROCounter, tsCounter])#display me!
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
          "Generated from Ruian's HIV model," ,
          'NSamples' = c(Gparams$NSamples),
          #'lambda' = lambda[lcounter],
          'max_size' = c(max_size)
        )
      settings_log <-
        paste0(
          'Sample generated at time: ',
          toString(lgd),
          #' with conditions and parameters: N = ',
          #toString(as.integer(Gparams$N)),
          # ', NGen = ',
          # toString(Gparams$NGen),
          ', NSamples = ',
          toString(Gparams$NSamples),
          #', lambda = ',
          #toString(lambda[lcounter]),
          ', max_size = ',
          toString(max_size),
          'used a non-sequential model'
        )
      
      filenameMat = paste(save_prefix,
        'TrainingSet_Kupperman',
        lgd,
        '-dsize-',
        toString(Gparams$NSamples),
        '.mat',
        sep = ''
      )#Make a time stamped output file name
      writeMat(
        filenameMat,
        matrices = data,
        labels = labels,
        values = values,
        indexes = indexes,
        settings_list = settings_list,
        log = settings_log,
        lambda = lambda,
        NPop=NPop
      )
      #access log with writeLines(log)
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
    #}
  }
  Gparams$filenames <- append(Gparams$filenames, filenames)
  
}

adjust_NGen <- function(mu, NGen){
  # Truncates NGen to 100 if mu>0
  if(mu>0){
    NGen <- 250
  } else {
    #do nothing
  }
  NGen
}
