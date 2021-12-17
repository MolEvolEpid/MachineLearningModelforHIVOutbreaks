shuffle_data <- function(infile,
                         Smethod = 'random',
                         Oname = 'Shuffled_TrainingSet_Kupperman',
                         prefix = 'Training_Data/',
                         ofix = 'Training_Data/') {
  # This script reorders the training data
  lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  library(R.matlab)
  file_name <- paste(prefix, infile, sep = '')


  thedata <- readMat(file_name)
  len <- dim(thedata$matrices)[1] #split on the first dimension #should be by matrix slice
  if (Smethod == 'random' || Smethod == 'rand')  # catch undocumented 'rand' option
  {
    indexlist <- sample(1:len, len, replace = FALSE)
  } else if (Smethod == 'reverse' || Smethod == 'rev') {
    indexlist <- seq(len, 1, by = -1)
  } else {
    rm(thedata)
    stop("The Smethod entered is invalid. Valid choices are 'random' and 'reverse'.")
  }
  dataout <- thedata
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
  message <- paste("The sample order was shuffled at time: ", lgd, sep = " ")
  settings_log <- paste(thedata$log, message, sep = "\n")
  filenameMat = paste(ofix, Oname, lgd, '-dsize-', toString(NSamples), '.mat', sep ='')  # Make a time stamped output file name

  writeMat(
    filenameMat,
    matrices = dataout$matrices,
    labels = dataout$labels,
    values = dataout$values,
    indexes = dataout$indexes,
    settings_list = thedata$settings_list,
    log = settings_log,
    NPop = dataout$NPop
  )  # access log with writeLines(log)
  print(filenameMat)

}
