#Data_Generator class file
#stores all attributes neccessary
#Defined with reference methods (RC)
# Data_Generator 1 ----------------

Data_Generator <- setRefClass(
  "Master_Params",
  # Fields ----------------
  fields = list(
    # use_sequential_model = "logical",
    # normalize_outputs = "logical",

    # #mutation between types rate parameters
    # mu_large = "numeric",
    # mu_small = "numeric",
    # mu_number = "numeric",
    #
    # #Fitness penalty parameters
    # theta_large = "numeric",
    # theta_small = "numeric",
    # theta_number = "numeric",

    #detectable mutation rate generation
    lambda_large = "numeric",
    lambda_small = "numeric",
    lambda_number = "numeric",

    #Selection parameter for exp model
    # s_large = "numeric",
    # s_small = "numeric",
    s_number = "numeric",
    
    s_mod_number = 'numeric',
    # s_step_fixed = 'logical',
    # include_s_zero = "logical",
    #force the inclusion of s=0

    #HIV unique simulation paramers
    use_linear_timesteps = "logical",
    ts_list_in = "vector",
    ts_min = "numeric",
    ts_max = "numeric",
    ts_num = "numeric",
    
    use_linear_RO = "logical",
    RO_list_in = "vector",
    RO_min = "numeric",
    RO_max = "numeric",
    RO_num = "numeric",
    
    #Generic simulation parameters
    N = "numeric",
    # NGen = "numeric",
    NSamples = "numeric",
    preseed_mutants = "logical",
    constant_s_step = "logical",
    
    filenames = "list"#To create a list of our output files

  ),
  # Methods ----------------
  methods = list(
    join_datasets =
      function (Oname = 'Training_Data/Joined_TrainingSet_KuppermanEtAl', prefix =
                  '') {
        lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")

        filenameMat = paste(prefix,
                            Oname,
                            lgd,
                            '-dsize-',
                            toString(NSamples),
                            '.mat',
                            sep = '')#Make a time stamped output file name

        #'Training_Data/'
        library(abind)
        library(R.matlab)
        flat_list <- unlist(filenames)
        if (length(flat_list) >= 2) {
          #don't do the loop if we dont have at least two datasets
          firstname <- paste(prefix, flat_list[1], sep = '')
          print(firstname)
          first <- readMat(firstname)
          #NSamples <- dim(first$matrices)[2]
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
            first$NPop <- append(first$NPop, second$NPop)
          }
          #lgd <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
          writeMat(
            filenameMat,
            matrices = first$matrices,
            labels = first$labels,
            values = first$values,
            indexes = first$indexes,
            settings_list = first$settings_list,
            log = first$log,
            NPop=first$NPop
          )#access log with writeLines(log)
          return(filenameMat)
        }
        print(paste0('A total of ', as.character(length(flat_list)), ' files were joined.'))
        filenameMat#return me
      },


    set_s_mod_number = function() {
      s_mod_number <<- s_number
    },

    logspace_mu = function() {
      if (mu_number > 1) {
        mu <-
          10 ** (seq(mu_large, mu_small, by = -(mu_large - mu_small) / (mu_number -
                                                                          1)))
      }
      else{
        mu <- 10 ** (mu_large)
      }
      return(mu)
    },
    logspace_s = function() {
      #default includes 0
      if (s_number > 1) {
        s <-
          (10 ** (seq(
            s_large, s_small, by = -(s_large - s_small) / (s_number - 1)
          ))) / N
        s_mod_number <<- s_number
      }
      else{
        s <- 10 ** (s_large) / N
        s_mod_number <<- s_number
      }
      return(s)
    },
    logspace_lambda = function() {
      if (lambda_number > 1) {
        lambda <-
          10 ** (seq(
            lambda_large,
            lambda_small,
            by = -(lambda_large - lambda_small) / (lambda_number - 1)
          ))
      }
      else{
        lambda <- 10 ** (lambda_large)
      }
      return(lambda)
    },
    logspace_theta = function() {
      if (theta_number > 1)
        theta <-
          10 ** (seq(
            theta_large,
            theta_small,
            by = -(theta_large - theta_small) / (theta_number - 1)
          ))
      else{
        theta <- 10 ** (theta_large)
      }
      return(theta)
    },
    values_timestep = function(){
      if(use_linear_timesteps){
        timesteps <- seq(min(ts_min, ts_max), max(ts_max, ts_min),by=(max(ts_min, ts_max) - min(ts_max, ts_min))/(ts_num) )
        return(timesteps)
      } else {
        ts_num <<- length(ts_list_in)
        return(ts_list_in)
      }
      
    },
    values_RO = function(){
      if(use_linear_RO){
        RO_steps <- seq(min(RO_min, RO_max), max(RO_max, RO_min),by=(max(RO_min, RO_max) - min(RO_max, RO_min))/(RO_num) )
        return(RO_steps)
      } else {
        RO_num <<- length(RO_list_in)
        return(RO_list_in)
      }
      
    },
    set_RO_num = function(){
      if(!use_linear_RO){
        RO_num <<- length(RO_list_in)
        if(!(RO_num>0)){
          stop("There must be at least one R0 value")
        }
      }
      return(RO_num)
    },
    set_ts_num = function(){
      if(!use_linear_timesteps){
        ts_num <<- length(ts_list_in)
        if(!(ts_num>0)){
          stop("There must be at least one timestep (ts) value")
        }
      }
      return(ts_num)
    }
    
  )
)


