#utility_functions


#####################################################################
#join_datasets
#Michael Kupperman
#start 7/7/2019
#Arguments: input
#       #...    - a list of file names to join (include .mat)
#       #prefix - a common prefix to append to all file names (folder location)
#       #Oname  - The output file name prefix (no date stamp) optional
#Arguments: output
#       #filenameMat -the file name of the saved file
#
#This function reads in a list of file names output by DataGen.R,
# and writes them into a new single .mat file
#####################################################################
join_datasets <-
  function (..., Oname = 'Joined_TrainingSet_Kupperman', prefix = 'Training_Data/') {
    library(abind)
    library(R.matlab)
    if (...length() >= 2) {
      #don't do the loop if we dont have at least two datasets
      firstname <- paste(prefix, ...elt(1), sep = '')
      print(firstname)
      first <- readMat(firstname)
      NSamples <- dim(first$matrices)[2]
      lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
      #print(dim(first$matrices))
      first$log <-
        paste(first$log,
              paste('Begin file joining event at time:', lgd, sep = ' '),
              sep = "\n")
      for (ii in 2:...length()) {
        second <-
          readMat(paste(prefix, ...elt(ii), sep = ''))#read in the next set of data
        #print(first$data)
        #print(second$data)
        print(ii)
        first$matrices <-
          abind(first$matrices, second$matrices, along = 1)
        first$indexes <- rbind(first$indexes, second$indexes)
        first$values <- rbind(first$values, second$values)
        first$labels <- append(first$labels, second$labels)
        first$log <-
          paste(first$log,
                paste('File merged with:', second$log, sep = ' '),
                sep = "\n")
      }
      #lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
      filenameMat = paste(prefix,
                          Oname,
                          lgd,
                          '-dsize-',
                          toString(NSamples),
                          '.mat',
                          sep = '')#Make a time stamped output file name
      writeMat(
        filenameMat,
        matrices = first$matrices,
        labels = first$labels,
        values = first$values,
        indexes = first$indexes,
        settings_list = first$settings_list,
        log = first$log
      )#access log with writeLines(log)
      filenameMat
    }
    print(paste0('A total of ', as.character(...length()), ' files were joined.'))
    print(filenameMat)
  }


join_list_datasets = #accepts a list of filenames, rather than them being individually entered.
  function (filenames,
            Oname = 'Training_Data/Joined_TrainingSet_Kupperman',
            prefix = '') {
    #'Training_Data/'
    library(abind)
    library(R.matlab)
    flat_list <- unlist(filenames)
    if (length(flat_list) >= 2) {
      #don't do the loop if we dont have at least two datasets
      firstname <- paste(prefix, flat_list[1], sep = '')
      print(firstname)
      first <- readMat(firstname)
      NSamples <- dim(first$matrices)[2]
      lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
      #print(dim(first$matrices))
      first$log <-
        paste(first$log,
              paste('Begin file joining event at time:', lgd, sep = ' '),
              sep = "\n")
      for (ii in 2:length(flat_list)) {
        second <-
          readMat(paste(prefix, flat_list[ii], sep = ''))#read in the next set of data
        #print(first$data)
        #print(second$data)
        print(ii)
        first$matrices <-
          abind(first$matrices, second$matrices, along = 1)
        first$indexes <- rbind(first$indexes, second$indexes)
        first$values <- rbind(first$values, second$values)
        first$labels <- append(first$labels, second$labels)
        first$log <-
          paste(first$log,
                paste('File merged with:', second$log, sep = ' '),
                sep = "\n")
      }
      #lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
      filenameMat = paste(prefix,
                          Oname,
                          lgd,
                          '-dsize-',
                          toString(NSamples),
                          '.mat',
                          sep = '')#Make a time stamped output file name
      writeMat(
        filenameMat,
        matrices = first$matrices,
        labels = first$labels,
        values = first$values,
        indexes = first$indexes,
        settings_list = first$settings_list,
        log = first$log
      )#access log with writeLines(log)
      filenameMat
    }
    print(paste0('A total of ', as.character(length(flat_list)), ' files were joined.'))
    filenameMat#return me
  }

#####################################################################
#sanitize_new_labels - UNDER DEVELOPMENT
#Michael Kupperman
#start 7/7/2019
#Arguments: input
#       #infile - a list of file names to join (include .mat)
#       #Oname  - The output file name prefix (no date stamp) optional
#Arguments: output
#       #filenameMat -the file name of the saved file
#
#This function reads in a file output by DataGen.R,
# and outputs a new file with new labels, first indexed by mu, then
#by theta. Note that labels will be overwritten for this new object.
#Note that rs (s) and theta are in the same position.
#####################################################################
sanitize_new_labels <-
  function (infile, Oname = 'CleanLabel_TrainingSet_Kupperman') {
    library(R.matlab)
    DataA <- readMat(infile)#read in the next set of data
    NSamples <- dim(DataA$matrices)[2]
    
    mu_list <- DataA$values[, 1]
    theta_list <- DataA$values[, 2]
    lambda_list <- DataA$values[, 3]
    
    unique_mu <- unique(mu_list)
    unique_theta <- unique(theta_list)
    unique_lambda <- unique(lambda_list)
    
    bins <-
      seq(1,
          to = length(unique_mu) * length(unique_theta),
          by = 1)#make a list of the number of possible pairs
    dim(bins) <- c(length(unique_theta), length(unique_mu))
    for (ii in 1:length(DataA$labels)) {
      index_mu <- match(mu_list[ii], unique_mu)
      index_theta <- match(theta_list[ii], unique_theta)
      #index_lambda <- match(lambda_list[ii],unique_lambda)
      labels[ii] = bins[index_theta, index_mu]
    }
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    message <-
      paste("New labels were written to the file at: ", lgd, sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat = paste(Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                          '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = DataA$matrices,
      labels = DataA$labels,
      values = DataA$values,
      indexes = DataA$indexes,
      settings_list = DataA$settings_list,
      log = DataA$settings_log
    )#access log with writeLines(log)
    print(filenameMat)
  }
#print(paste('A total of ', str(int(...length)), 'files were joined',sep=' '))


#####################################################################
#Trim_dataset_samples
#Michael Kupperman
#start 7/7/2019
#Arguments: input
#       #infile - a file name (include .mat), this file contains a $matrices
#       #NewSize- An integer less than the size of the matrix to reduce to
#       #Oname  - The output file name prefix (no date stamp) optional
#Arguments: output
#       #filenameMat -the file name of the saved file
#
#This function reads in a file output by DataGen.R, and then reduces
#the size of the pairwise matrices defined by NewSize, and saves the
#result in
#####################################################################
Trim_dataset_samples <-
  function (infile,
            NewSize = 10,
            Oname = 'Trimmed_TrainingSet_Kupperman') {
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    
    library(R.matlab)
    DataA <- readMat(infile)#read in the next set of data
    
    mu_list <- DataA$values[, 1]
    if (NewSize > dim(DataA$matrices)[2] ||
        NewSize > dim(DataA$matrices)[3]) {
      stop("The reduced size is larger than the original size. Choose a different NewSize.")
    }
    NewArray <-
      array(0, dim = c(dim(DataA$matrices)[1], NewSize, NewSize))
    #print(dim(NewArray))
    print(dim(DataA$matrices)[1])
    for (ii in 1:dim(DataA$matrices)[1]) {
      NewArray[ii, ,] <-
        DataA$matrices[ii, 1:NewSize, 1:NewSize]#get the upper left region only
    }
    message <-
      paste("The top ",
            str(NewSize),
            "entries of each matrix was trimmed at: ",
            lgd,
            sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat = paste(Oname, lgd, '-dsize-', toString(NewSize), '.mat', sep =
                          '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = NewArray,
      labels = DataA$labels,
      values = DataA$values,
      indexes = DataA$indexes,
      settings_list = DataA$settings_list,
      log = DataA$settings_log
    )
    filenameMat#return me
  }
#print(paste('A total of ', str(int(...length)), 'files were joined',sep=' '))

#####################################################################
#data_corners
#Michael Kupperman
#start 7/7/2019
#Arguments: input
#       #infile - a file name (include .mat), this file contains a $matrices
#       #Oname  - The output file name prefix (no date stamp) optional
#Arguments: output
#       #filenameMat -the file name of the saved file
#
#This function reads in a file output by DataGen.R, and then returns
#a new datafile of the same structure with only the (mu, theta) corners
#kept. All intermediate points are removed.
#####################################################################
data_corners <-
  function(infile, Oname = 'Corners_TrainingSet_Kupperman') {
    library(abind)
    library(R.matlab)
    DataA <- readMat(infile)#read in the next set of data
    NSamples <- dim(DataA$matrices)[2]
    
    mu_list <- DataA$values[, 1]
    theta_list <- DataA$values[, 2]
    unique_mu <- unique(mu_list)
    unique_theta <- unique(theta_list)
    
    array_length <-
      dim(DataA$matrices)[1] / (length(unique_theta) * length(unique_mu)) * 4
    Nlabels <-
      matrix(0, array_length, 1) #For one-hot encodings of bins
    Nvalues <-
      matrix(0, array_length , 2) #save the parameter values (mu and theta)
    Nindexes <-
      matrix(0, array_length, 2) #save the muCounter & thetaCounter indices (i.e. the grid coordinates)
    
    min_mu <- min(unique_mu)
    max_mu <- max(unique_mu)
    
    min_theta <- min(unique_theta)
    max_theta <- max(unique_theta)
    bins <- seq(1, 4)
    dim(bins) <- c(2, 2)#only need 2 points per direction
    
    nn = 1
    newmatrices <-
      array(dim = c(array_length, NSamples, NSamples))#preallocate for speed
    for (ii in 1:length(DataA$labels)) {
      if ((DataA$values[ii, 1] == min_mu ||
           DataA$values[ii, 1] == max_mu) &&
          (DataA$values[ii, 2] == min_theta ||
           DataA$values[ii, 2] == max_theta)) {
        newmatrices[nn, ,] <- DataA$matrices[ii, ,]
        
        if (DataA$values[ii, 1] == min_mu) {
          mu_iter <- 1
        }
        else{
          mu_iter <- 2
        }
        if (DataA$values[ii, 2] == min_theta) {
          theta_iter <- 1
        }
        else{
          theta_iter <- 2
        }
        Nindexes[nn, 1] <- mu_iter
        Nindexes[nn, 2] <- theta_iter
        Nlabels[nn] <- bins[mu_iter, theta_iter]
        Nvalues[nn, 1] <- DataA$values[ii, 1]
        Nvalues[nn, 2] <- DataA$values[ii, 2]
        
        nn = nn + 1#go to next slice
      }
    }
    
    
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    message <-
      paste("Only the corners mu and theta were kept. This occured at: ",
            lgd,
            sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat = paste(Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                          '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = newmatrices,
      labels = Nlabels,
      values = Nvalues,
      indexes = Nindexes,
      settings_list = DataA$settings_list,
      log = settings_log
    )#access log with writeLines(log)
    filenameMat
  }

#####################################################################
#batch_join_data_corners
#Michael Kupperman
#start 7/12/2019
#Arguments: input
#       # ... - a list of file names (include .mat), this file contains a $matrices
#Arguments: output
#       #filenameMat -the file name of the saved file
#
#This function reads in a file output by DataGen.R, and then returns
#a new datafile of the same structure with only the (mu, theta) corners
#kept. All intermediate points are removed.
#####################################################################
#A batch script for the prior function
batch_join_data_corners <- function(...) {
  filenames <- list()
  for (ii in 1:...length()) {
    print(ii)
    fnam <- data_corners(...elt(ii))
    filenames <- append(filenames, fnam)
    Sys.sleep(1)
  }
  join_datasets(filenames)
}

#UNDER DEVELOPMENT
#Rescales the data in each matrix by dividing  by the largest element in the matrix
rescale_data <-
  function(infile,
           Oname = 'Scaled_TrainingSet_Kupperman',
           prefix = 'Training_Data/',
           ofix = 'Training_Data/') {
    library(R.matlab)
    file_name <- paste(prefix, infile, sep = '')
    print(file_name)
    DataA <- readMat(file_name)
    NSamples <- dim(DataA$matrices)[2]
    #read in the next set of data
    out_array <-
      array(0, dim = dim(DataA$matrices))#create an empty array
    for (ii in 1:dim(DataA$matrices)[1]) {
      divisor <- (max(DataA$matrices[ii, , ]))
      if (divisor)
        out_array[ii, , ] <- (DataA$matrices[ii, , ]) / divisor
    }
    
    #else{warning(A matrix of zeros was encountered)}
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    message <-
      paste("Data was rescaled to max(matrix)=1 at: ", lgd, sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat = paste(ofix, Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                          '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = out_array,
      labels = DataA$labels,
      values = DataA$values,
      indexes = DataA$indexes,
      settings_list = DataA$settings_list,
      log = DataA$settings_log
    )#access log with writeLines(log)
    filenameMat#return me
  }


#UNDER DEVELOPMENT
rescale_data2 <-
  function(infile, Oname = 'Scaled_TrainingSet_Kupperman') {
    library(R.matlab)
    DataA <- readMat(infile)#read in the next set of data
    NSamples <- dim(DataA$matrices)[2]
    mu_list <- DataA$values[, 1]
    theta_list <- DataA$values[, 2]
    unique_mu <- unique(mu_list)
    unique_theta <- unique(theta_list)
    bins <-
      seq(1,
          to = length(unique_mu) * length(unique_theta),
          by = 1)#make a list of the number of possible pairs
    dim(bins) <- c(length(unique_theta), length(unique_mu))
    for (ii in 1:length(DataA$labels)) {
      index_mu <- match(mu_list[ii], unique_mu)
      index_theta <- match(theta_list[ii], unique_theta)
      labels[ii] = bins[index_theta, index_mu]
    }
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    message <-
      paste("New labels were written to the file at: ", lgd, sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat <-
      paste(Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
              '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = DataA$matrices,
      labels = DataA$labels,
      values = DataA$values,
      indexes = DataA$indexes,
      settings_list = DataA$settings_list,
      log = DataA$settings_log
    )#access log with writeLines(log)
  }

list_from_master <- function(directory = 'Training_Data') {
  dirlist <- list.files(paste0('./', directory))
  dirlen <- length(dirlist)
  outchar_1 <- ''
  outchar_2 <- ''
  outchar_3 <- ''
  
  for (ii in 2:dirlen) {
    #outchar_1 <-
    
  }
}


# Select a feature or set of features out of a dataset.
filter_feature <- function(infile,
                           features = list(),
                           #enter as a list
                           Oname = 'Shuffled_TrainingSet_Kupperman',
                           prefix = 'Training_Data/',
                           ofix = 'Training_Data/') {
  library(R.matlab)
  file_name <- paste(prefix, infile, sep = '')
  
  #This script generates the training data in a new order
  lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  
  library(R.matlab)
  thedata <- readMat(file_name)
  len <-
    dim(thedata$matrices)[1]#split on the first dimension #should be by matrix slice
  if (length(features)>0)
  {
    positions = list()
    for (feat in features){
      positions <- append(positions, which(thedata$labels == feat) )
    }
    positions <- (unlist(positions))
    #print(positions)
    
    len <- length(positions)
  } else {
    rm(thedata)
    stop("The features list is empty or not recognized.")
  }
  N <- dim(thedata$matrices)[2]
  
  #dataout <- thedata#data.frame(thedata$matrices, thedata$labels)
  dataout <- list('matrices' = array(0, dim = c(len, N, N)), 
                  'labels' = (unlist(rep(list(0),len))),
                  'values' = array(0, dim = c(len, dim(thedata$values)[2])),
                  'indexes' = array(0, dim = c(len, dim(thedata$indexes)[2]))
                  )
  #dim(dataout$matrices) <- array(0, dim = c(len, N, N))  
  #dim(dataout$labels) <- list(unlist(rep(list(0),len)))
  #dim(dataout$values) <- array(0, dim = c(len, dim(thedata$values)[2]))
  #dim(dataout$indexes) <-array(0,dim = c(len, dim(thedata$indexes)[2]))
  
  for (i in 1:dim(dataout$matrices)[1]) {
    for (j in 1:dim(dataout$matrices)[2]) {
      for (k in 1:dim(dataout$matrices)[3]) {
        dataout$matrices[i, j, k] <- thedata$matrices[positions[i], j, k]
      }
    }
    dataout$labels[i] <- thedata$labels[positions[i]]
    dataout$values[i,] <- thedata$values[positions[i],]
    dataout$indexes[i,] <- thedata$indexes[positions[i],]
  }
  
  NSamples <- dim(dataout$matrices)[2]
  message <-
    paste("The set was downselected for features: ", features,  " at time: ", lgd, sep = " ")
  settings_log <- paste(thedata$log, message, sep = "\n")
  filenameMat = paste(ofix, Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                        '')#Make a time stamped output file name
  
  
  writeMat(
    filenameMat,
    matrices = dataout$matrices,
    labels = dataout$labels,
    values = dataout$values,
    indexes = dataout$indexes,
    settings_list = thedata$settings_list,
    log = settings_log
  )#access log with writeLines(log)
  print(filenameMat)
  
}


shuffle_data <- function(infile,
                         Smethod = 'random',
                         Oname = 'Shuffled_TrainingSet_Kupperman',
                         prefix = 'Training_Data/',
                         ofix = 'Training_Data/') {
  library(R.matlab)
  file_name <- paste(prefix, infile, sep = '')
  
  #This script generates the training data in a new order
  lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  
  library(R.matlab)
  thedata <- readMat(file_name)
  len <-
    dim(thedata$matrices)[1]#split on the first dimension #should be by matrix slice
  if (Smethod == 'random' || Smethod == 'rand')
  {
    indexlist <- sample(1:len, len, replace = FALSE)
  } else if (Smethod == 'reverse' || Smethod == 'rev') {
    indexlist <- seq(len, 1, by = -1)
  } else {
    rm(thedata)
    stop("The Smethod entered is invalid. Valid choices are 'random' and 'reverse'.")
  }
  #dataout<- df(matricies = matrix(0,dim(thedata$matrices)), labels=matrix(0,dim(thedata$labels)))
  dataout <- thedata#data.frame(thedata$matrices, thedata$labels)
  for (i in 1:dim(thedata$matrices)[1]) {
    for (j in 1:dim(thedata$matrices)[2]) {
      for (k in 1:dim(thedata$matrices)[3]) {
        dataout$matrices[i, j, k] <- thedata$matrices[indexlist[i], j, k]
      }
    }
    dataout$labels[i] <- thedata$labels[indexlist[i]]
    dataout$values[i,] <- thedata$values[indexlist[i],]
    dataout$indexes[i,] <- thedata$indexes[indexlist[i],]
    dataout$NPop[i] <- thedata$NPop[indexlist[i]]
    
  }
  NSamples <- dim(dataout$matrices)[2]
  message <-
    paste("The sample order was shuffled at time: ", lgd, sep = " ")
  settings_log <- paste(thedata$log, message, sep = "\n")
  filenameMat = paste(ofix, Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                        '')#Make a time stamped output file name
  
  
  writeMat(
    filenameMat,
    matrices = dataout$matrices,
    labels = dataout$labels,
    values = dataout$values,
    indexes = dataout$indexes,
    settings_list = thedata$settings_list,
    log = settings_log,
    NPop = dataout$NPop
  )#access log with writeLines(log)
  print(filenameMat)
  
}

#AncestoryPlot
#AncestryPlot(Parents, 1)
AncestryPlot <- function(Parents,
                         IndivStart = 1,
                         GenStart = 1) {
  #Parents is a NGen x NSamples ancestry matrix - each position should encode the parent of that member of the population
  if (length(dim(Parents)) != 2) {
    stop("The parents matrix is not of the correct shape")
  }
  if (GenStart >= dim(Parents)[2])
  {
    stop("The generation entered (GenStart) exceeds the generations stored in Parents")
  }
  if (IndivStart > dim(Parents)[1])
  {
    stop(
      "The individual number entered (IndivStart) exceeds the number of individuals in Parents"
    )
  }
  
  lv <- (dim(Parents)[2] - GenStart - 1)
  print(lv)
  compo <-
    vector(mode = "numeric", length = lv)
  print(length(compo))
  N = dim(Parents)[1]
  NSamples = N
  NGen = dim(Parents)[2]
  
  compo[1] = 1 / NSamples #default to one individual in the population
  ancestor = list(IndivStart)
  
  ii = 2
  seqobj = (seq(GenStart + 1, dim(Parents)[2], by = 1))
  print(seqobj)
  for (index in seqobj) {
    av = NULL
    ancestor_new = NULL
    for (Sancestor in ancestor) {
      av = which(Sancestor == Parents[, index])
      ancestor_new = append(ancestor_new, av)
    }
    compo[ii] = length(ancestor_new) / NSamples
    ancestor = ancestor_new
    ii = ii + 1
    
  }
  seqobj = append(GenStart, seqobj)#now we put it back
  print(length(seqobj))
  print(length(compo))
  plot(seqobj, compo)
}

AncestryPlot2 <- function(Parents,
                          GenStart = 1) {
  #This plots all ancestries starting in the desired generation, colorized
  #Parents is a NGen x NSamples ancestry matrix - each position should encode the parent of that member of the population
  if (length(dim(Parents)) != 2) {
    stop("The parents matrix is not of the correct shape")
  }
  if (GenStart >= dim(Parents)[2])
  {
    stop("The generation entered (GenStart) exceeds the generations stored in Parents")
  }
  N = dim(Parents)[1]
  NGen = dim(Parents)[2]
  
  series = vector("list", length = N)
  for (indiv in 1:dim(Parents)[1]) {
    lv <- (dim(Parents)[2] - GenStart - 1)
    #print(lv)
    compo <-
      vector(mode = "numeric", length = lv)
    #print(length(compo))
    
    compo[1] = 1 / N #default to one individual in the population
    ancestor = list(indiv)
    
    ii = 2
    seqobj = (seq(GenStart + 1, dim(Parents)[2], by = 1))
    #print(seqobj)
    for (index in seqobj) {
      av = NULL
      ancestor_new = NULL
      for (Sancestor in ancestor) {
        #if (length(ancestor)>0){
        av = which(Sancestor == Parents[, index])
        ancestor_new = append(ancestor_new, av)
      }
      compo[ii] = length(ancestor_new) / N
      ancestor = ancestor_new
      ii = ii + 1
      #}
    }
    series[[indiv]] = compo
    
    seqobj = append(GenStart, seqobj)#now we put it back
  }
  matplot(
    seqobj,
    do.call(cbind, series),
    xlim = c(1, NGen),
    ylim = c(0, 1),
    type = "l",
    lty = 1
  )
}

AncestryPlot3 <- function(Parents,
                          GenStart = 1, individuals = seq(1,10)) {
  #This plots the ancestries of 'individuals' starting in the desired generation, colorized
  #Parents is a NGen x NSamples ancestry matrix - each position should encode the parent of that member of the population
  if (length(dim(Parents)) != 2) {
    stop("The parents matrix is not of the correct shape")
  }
  if (GenStart >= dim(Parents)[2])
  {
    stop("The generation entered (GenStart) exceeds the generations stored in Parents")
  }
  print(length(individuals))
  print(individuals)
  if (length(individuals) > dim(Parents)[1])
  {
    stop("There are more individuals to track than there are members in the population")
  }
  N = dim(Parents)[1]
  NGen = dim(Parents)[2]
  series = vector("list", length = N)
  for (indiv in individuals) {
    indiv_index = which(indiv == individuals)
    #dim(Parents)[1]
    lv <- (dim(Parents)[2] - GenStart - 1)
    #print(lv)
    compo <-
      vector(mode = "numeric", length = lv)
    #print(length(compo))
    
    compo[1] = 1 / N #default to one individual in the population
    ancestor = list(indiv)
    
    ii = 2
    seqobj = (seq(GenStart + 1, dim(Parents)[2], by = 1))
    #print(seqobj)
    for (index in seqobj) {
      av = NULL
      ancestor_new = NULL
      for (Sancestor in ancestor) {
        #if (length(ancestor)>0){
        av = which(Sancestor == Parents[, index])
        ancestor_new = append(ancestor_new, av)
      }
      compo[ii] = length(ancestor_new) / N
      ancestor = ancestor_new
      ii = ii + 1
      #}
    }
    series[[indiv_index]] = compo
    
    seqobj = append(GenStart, seqobj)#now we put it back
  }
  matplot(
    seqobj,
    do.call(cbind, series),
    xlim = c(1, NGen),
    #ylim = c(0, 1),
    type = "l",
    lty = 1
  )
}



Profiler <- function() {
  library(profvis)
  profvis({
    Parents = generate_pd_matrix_ns(500, 500, 10, .1, .1, .1, 3)
    AncestryPlot2(Parents)
  })
}




mutationPlot <- function(mutations, mutN = 1) {
  if (length(dim(mutations)) != 2) {
    stop("The parents matrix is not of the correct shape")
  }
  N <- dim(mutations)[1]
  NGen <- dim(mutations)[2]
  lv <- (NGen - 1)
  compo <- vector(mode = "numeric", length = lv)
  last <- matrix(0, nrow = N, ncol = NGen)
  
  for (ii in 1:N) {
    for (jj in 1:NGen) {
      #print(last[ii, jj])
      if (is.null((tail(mutations[[ii, jj]], 1)))) {
        last[ii, jj] <- 0
      }
      else{
        last[ii, jj] <- unlist(tail(mutations[[ii, jj]], 1))
      }
    }
  }
  for (jj in 1:lv) {
    compo[jj] = sum(last[, jj] == mutN) / N
  }
  plot(
    compo,
    xlim = c(1, NGen),
    ylim = c(0, 1),
    type = "l",
    lty = 1
  )
}

mutationPlotAll <- function(mutations, mutN = NULL)
  #Legacy support for mutN argument
{
  if (length(dim(mutations)) != 2)
  {
    stop("The parents matrix is not of the correct shape")
  }
  N <- dim(mutations)[1]
  #print("N is ")
  #print(N)
  series = vector("list", length = N)
  NGen <- dim(mutations)[2]
  lv <- (NGen - 1)
  
  compo <- vector(mode = "numeric", length = lv)
  last <- matrix(0, nrow = N, ncol = NGen)
  maxmutant <-
    max(unlist(lapply(mutations, length)))#get the maximum number of elements in a list
  
  for (ii in 1:N)
  {
    for (jj in 1:NGen)
    {
      #print(last[ii, jj])
      if (is.null((tail(mutations[[ii, jj]], 1))))
      {
        last[ii, jj] <- 0
      }
      else
      {
        last[ii, jj] <- unlist(tail(mutations[[ii, jj]], 1))
      }
    }
  }
  for (mutation in 1:maxmutant) {
    for (jj in 1:lv)
    {
      compo[jj] <- sum(last[, jj] == mutN) / N
      print(compo[jj])
    }
    series[[mutation]] <- compo
  }
  yvals <- do.call(cbind, series)
  print(yvals)
  print("yvals dim is")
  print(dim(yvals))
  # matplot(
  #   1:NGen,
  #   yvals,
  #   xlim = c(1, NGen),
  #   ylim = c(0, 1),
  #   type = "l",
  #   col = 1:50,
  #   lty = 1,
  #   xlab = "Generation",
  #   ylab = "Fraction of population decended from an individual"
  # )
  yvals
}

#Rescales the data in each matrix by dividing  by the largest element in the matrix
cluster_matrices <-
  function(infile,
           Oname = 'Clustered_TrainingSet_Kupperman',
           prefix = 'Training_Data/',
           ofix = 'Training_Data/',
           alg = "HC_ward") {
    library(R.matlab)
    file_name <- paste(prefix, infile, sep = '')
    print(file_name)
    DataA <- readMat(file_name)
    NSamples <- dim(DataA$matrices)[2]
    #read in the next set of data
    out_array <-
      array(0, dim = dim(DataA$matrices))#create an empty array
    for (ii in 1:dim(DataA$matrices)[1]) {
      distmat <- DataA$matrices[ii, , ]
      dim(distmat) <- c(NSamples, NSamples)
      distobj <- as.dist(distmat)
      out_array[ii, , ] <-
        as.matrix(permute(distobj, seriate(x = distobj, method = alg)))
      
      #computes the new dist, then builds matrix
    }
    
    #else{warning(A matrix of zeros was encountered)}
    lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
    message <-
      paste("Data was rescaled to max(matrix)=1 at: ", lgd, sep = " ")
    settings_log <- paste(DataA$log, message, sep = "\n")
    
    filenameMat = paste(ofix, Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep =
                          '')#Make a time stamped output file name
    writeMat(
      filenameMat,
      matrices = out_array,
      labels = DataA$labels,
      values = DataA$values,
      indexes = DataA$indexes,
      settings_list = DataA$settings_list,
      log = DataA$settings_log
    )#access log with writeLines(log)
    filenameMat#return me
  }

#resort_data(matlist)
