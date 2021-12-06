#rm(list = ls())
#library(plotly)
#install.packages("R.matlab")
library(abind)
library(R.matlab)
# library(seriation)
#set.seed(1000)


gen_Geneology_exp <- function (NSamples, R0, NPop, totalSteps) {
  #prob_death=0.05
  #prob_birth=R0/24 #prob_death*R0
  
  N = NPop
  
  Parents = matrix(0, max(totalSteps, 12) * N, 2)
  
  pp = 2
  currStep = 1
  active = c(1)
  birthStep = c(1)
  endStep = sample(13:36, 1)
  #print (N)
  flag = TRUE
  startstep = 1000
  while (currStep < totalSteps + startstep) {
    #print (active)
    
    indsRem = NULL
    
    L_active = length(active)
    if (flag && L_active > 0.9 * N) {
      flag = FALSE
      startstep = currStep
      # print (startstep)
    }
    
    
    L_active = length(active)
    # nBirths = min(N - L_active, rbinom(1, L_active, R0 / 24))
    #print (nBirths)
    if (N - L_active > 0) {
      offsprings = (currStep - birthStep<3) * 0.4/3 * R0 / 0.505 + (currStep-birthStep>=3) * 0.005 * R0/0.505
      tot_offsprings=ceiling(min(N - L_active, sum(offsprings)))
      probs=offsprings/sum(offsprings)
      
      prts = sample(1:L_active, tot_offsprings, replace = TRUE)
      Parents[pp:(pp + tot_offsprings - 1), 1] = active[prts]
      Parents[pp:(pp + tot_offsprings - 1), 2] = currStep
      active = c(active, pp:(pp + tot_offsprings - 1))
      birthStep = c(birthStep, rep(currStep+1, tot_offsprings))
      endStep = c(endStep, currStep + sample(13:36, tot_offsprings, replace =
                                               TRUE))
    }

    indsRem = which(endStep == currStep)
    len_Rem = length(indsRem)
    if (len_Rem > 0 && len_Rem < L_active) {
      active = active[-indsRem]
      endStep = endStep[-indsRem]
      birthStep =birthStep[-indsRem]
    }
    
    currStep = currStep + 1
    pp = pp + tot_offsprings
    #print (active)
    #print (endStep)
  }

  #hist(Geneology[,2])
  #print (Parents[1:pp,])
  #print (Parents[c(1838,1456,518),])
  #print (tail(Parents))
  
  #print (length(active))
  Samples = sample(active, NSamples, replace = FALSE)
  #print (c(pp,length(active),currStep))
  #print (Samples)
  
  Geneology <- matrix(0, 2 * NSamples, 5)
  Geneology[1:(2 * NSamples - 1), 1] <- c(1:(2 * NSamples - 1))
  
  #Geneology[1:NSamples,3]=currStep
  ####Taking into account of the sampling time
  #Parents[Samples,2]=Parents[Samples,2]+rpois(NSamples,4)
  #Geneology[1:NSamples,3]=Parents[Samples,2]+rpois(NSamples,6)
  Geneology[1:NSamples, 3] = currStep + rpois(NSamples, 6)
  nNodes = NSamples
  IDs = 1:NSamples
  pp = NSamples + 1
  
  #print (Parents[c(1838,1456,518),])
  while (nNodes > 1) {
    ind = which.max(Parents[Samples, 2])
    #print (Samples)
    #print (Parents[Samples,])
    
    parent = Parents[Samples[ind], 1]
    #print (parent)
    
    coall = which(Samples == parent)
    if (length(coall) == 0) {
      Samples[ind] = parent
    } else{
      Geneology[pp, 3] = Parents[Samples[ind], 2]
      
      Geneology[IDs[ind], 2] = pp
      Geneology[IDs[coall], 2] = pp
      if (Geneology[IDs[coall], 3] < Parents[Samples[ind], 2])
        Geneology[IDs[coall], 3] = Parents[Samples[ind], 2]
      
      IDs[coall] = pp
      Samples = Samples[-ind]
      IDs = IDs[-ind]
      pp = pp + 1
      nNodes = nNodes - 1
    }
  }
  #print(Geneology)
  ####
  inds = 1:(2 * NSamples - 1)
  Geneology[inds, 3] = Geneology[inds, 3] - Geneology[(2 * NSamples - 1), 3]
  Geneology[inds, 3] = max(Geneology[inds, 3]) - Geneology[inds, 3]
  
  inds = 1:(2 * NSamples - 2)
  Geneology[inds, 4] = Geneology[Geneology[inds, 2], 3] - Geneology[inds, 3]
  Geneology[inds, 5] = Geneology[inds, 4] / (sum(Geneology[inds, 4]))
  #print (Geneology)
  return (Geneology)
}
gen_Geneologies <- function (NSamples, R0, NPop, totalSteps) {
  trees = list()
  Geneology = gen_Geneology_exp(NSamples, R0, NPop, totalSteps)
  trees[[1]] = Geneology
  return (trees)
}

Pairwise_diff_HIV <- function(Geneology, spike_root=FALSE) {
  # Convert a geneology to a pairwise difference matrix
  # spike_root option adds a row/column for a node at the root
  
  M = length(Geneology[, 1]) / 2
  
  if (spike_root) {
    M <- M + 1
  }  # 1 more root node
  mrca_mat <- matrix(0, M, M)
  pair_diff_mat <-  matrix(0, M, M)
  ind_par_mat <- Geneology
  if (spike_root){ M <- M - 1}
  #print (ind_par_list)
  ## THESE FOR LOOPS WILL GET US A MRCA MATRIX (YAY)
  for (k in 1:(M - 1)) {
    ancestors = rep(1e+5, M)
    pp = k
    ii = 1
    while (pp != 0) {
      ancestors[ii] = pp
      ii = ii + 1
      pp = Geneology[pp, 2]
    }
    for (l in (k + 1):M) {
      pp = l
      
      while (!(pp %in% ancestors)) {
        pp = Geneology[pp, 2]
      }
      
      mrca_mat[k, l] <- pp
      mrca_mat[l, k] <- pp
    }
  }
  if(spike_root){
    oldest_node = length(Geneology[,1]) - 1
    M <- M + 1
    mrca_mat[M,] = oldest_node
    mrca_mat[,M] = oldest_node
    mrca_mat[M,M] = 0  # 
  }
  #print (mrca_mat)
  # Uses MRCA matrix to sum up total mutations between any two ind.
  for (k in 1:(M - 1)) {
    for (l in (k + 1):M) {
      finally <- mrca_mat[k, l]
      
      holder <- ind_par_mat[k, 2]
      sumhold <- ind_par_mat[k, 5]
      while (holder != finally) {
        sumhold <- sumhold + ind_par_mat[holder, 5]
        holder <- ind_par_mat[holder, 2]
        
      }
      
      holder <- ind_par_mat[l, 2]
      sumhold <- sumhold + ind_par_mat[l, 5]
      while (holder != finally) {
        sumhold <- sumhold + ind_par_mat[holder, 5]
        holder <- ind_par_mat[holder, 2]
        
      }
      pair_diff_mat[k, l] <- sumhold
      pair_diff_mat[l, k] <- sumhold
    }
  }
  #dd = as.dist(pair_diff_mat)
  #hhc = hclust(dd, method = 'centroid')   ### clustering
  #sorted_hhc = reorder(hhc, dd, method="OLO")
  #pair_diff_mat_sorted = pair_diff_mat[sorted_hhc$order, sorted_hhc$order]
  return(pair_diff_mat)
}


gen_trees_matrices <- function(Geneologies, Nmuts, spike_root=FALSE) {
  if(!is.vector(Geneologies)){
    Geneologies = list(Geneologies)
  }
  NGeneologies = length(Geneologies)
  NSamples = length(Geneologies[[1]][, 1]) / 2
  
  Trees = list()
  # This uses non-type-safe addition: int + bool = int
  Matrices = array(NA, dim = c(1, NSamples + spike_root, NSamples + spike_root))
  
  id = 1
  for (ii in 1:NGeneologies) {
    if (!is.null(Geneologies[[ii]])) {
      Geneology = Geneologies[[ii]]
      inds = 1:(2 * NSamples - 2)
      #print (Geneology)
      #Geneology[inds,5]=rmultinom(1,Nmuts,prob=Geneology[inds,5])
      Geneology[inds, 5] = rpois(length(inds), Geneology[inds, 4] * Nmuts)
      res = Pairwise_diff_HIV(Geneology, spike_root=spike_root)
      #print (res)
      #if (max(res)>0)  Matrices[id,,]=res/max(res)
      #else Matrices[id,,]=res
      Matrices[id, ,] = res
      #if (sum(is.na(Matrices[id,,])>0)) {print (res);print (max(res)>0)}
      
      Trees[[id]] = Geneology
      
      #labels[id]=gen_label(Geneology,Geneologies)
      id = id + 1
    }
  }
  return(list(
    Matrices = Matrices,
    labels = labels,
    Trees = Trees
  ))
  
}


HIV_sim <- function(NSamples = 20, NumRep = 3) {
  #if(0){
  NSamples = 20
  
  NumRep = 3
  
  
  #R0s=c(seq(1.2,1.8,by=0.2),rep(2:6))
  R0s = c(5, 15)
  
  TotalSteps = 12 * c(0, 1, 2, 10)
  
  #R0s=2;totalSteps=24*20;Nmut=2
  print (length(R0s) * length(TotalSteps))
  TrainingSet = list(Matrices = NULL, labels = NULL)
  
  par(mfrow = c(2, 4), mar = c(2, 2, 0.1, 0.1))#c(length(R0s),length(TotalSteps)),
  
  for (jj in 1:length(R0s)) {
    R0 = R0s[jj]
    for (kk in 1:length(TotalSteps)) {
      totalStep = TotalSteps[kk]
      print (c(R0, totalStep))
      
      TREEs = list(NULL)
      tp = 1
      for (ii in 1:NumRep) {
        #Nmut=runif(1,min=0.001,max=0.005)*300/12
        Nmut = 0.002 * 300 / 12#don't change me
        NPop = floor(10 ^ runif(1, min = 3, max = 4)) # rand uniform, NOT run if
        Geneologies = gen_Geneologies(NSamples, R0, NPop, totalStep)
        #print ('yes')
        out = gen_trees_matrices(Geneologies, Nmut)
        #print ('yes')
        
        TrainingSet$Matrices = abind(TrainingSet$Matrices, out$Matrices, along =
                                       1)
        
        TREEs[tp] = out$Trees
        tp = tp + 1
      }
      TrainingSet$labels = abind(TrainingSet$labels,
                                 matrix(c(jj, kk), NumRep, 2, byrow = TRUE),
                                 along = 1)
      
      if (0) {
        # generate images
        for (ii in 1:length(TREEs)) {
          if (ii == 1)
            plot(TREEs[[ii]][1:(2 * NSamples - 1), 3])
          else
            points(TREEs[[ii]][1:(2 * NSamples - 1), 3])
        }
      }
        # text(15, 0.9, paste('R0=', R0, ' Years=', totalStep / 12))
        
        dd = as.dist(out$Matrices[1, ,])
        hhc = hclust(dd, method = 'centroid')   ### clustering
        sorted_hhc = reorder(hhc, dd, method="OLO")
        verbose=FALSE
        if (verbose == TRUE){
          print(hhc$order)
          plot(hhc)
          print(sorted_hhc$order)
          plot(sorted_hhc)
          col = rev(heat.colors(999))
          stats::heatmap(out$Matrices[1,sorted_hhc$order, sorted_hhc$order], symm = TRUE, Rowv=NA, Colv = "Rowv", col=col)
          stats::heatmap(out$Matrices[1,hhc$order, hhc$order], symm = TRUE, Rowv=NA, Colv = "Rowv", col=col)
        }        
        out$Matrices = out$Matrices[,sorted_hhc$order, sorted_hhc$order, drop=FALSE]
        print(stats::heatmap(out$Matrices[,,], symm = TRUE, Rowv=NA, Colv="Rowv"))
      
    return(out$Matrices)      
    }
  }
  
  print (dim(TrainingSet$Matrices))
  filenameMat = 'HIV_Train/TrainingSet_Exp_20.mat'
  writeMat(filenameMat,
           matrices = TrainingSet$Matrices,
           labels = TrainingSet$labels)
}

generate_pd_matrix_HIV <- function(NSamples = 20,
                                   R0,
                                   totalStep,
                                   spike_root=FALSE) {
  # spike_root adds an extra row/column to the evolutionary pairwise distance matrix
  # for the distance from the root node.
  Nmut = 0.002 * 300 / 12#don't change me
  NPop = floor(10 ^ runif(1, min = 3, max = 4))
  Geneologies = gen_Geneologies(NSamples, R0, NPop, 12 * totalStep) #here's the workhorse
  out = gen_trees_matrices(Geneologies, Nmut, spike_root=spike_root)
  out = list("Matrices"=out$Matrices, "NPop"=NPop)
  return(out) # $Matrices
}
# tmp = generate_mega_pd_matrix_HIV(NSamples = cluster_sample_size, R0 = R0_default, random_R0 = randomize_R0_vals, clusters=number_of_clusters, shuffle=shuffle_in_cluster)

generate_mega_pd_matrix_HIV <- function(NSamples,
                                       R0, random_R0=TRUE,
                                       clusters = 3, shuffle=TRUE){
  spike_root=TRUE
  Matrices = array(NA, dim = c(clusters, NSamples + spike_root, NSamples + spike_root))
  totalStep_values = sample(x=c(0,2,10), size=clusters, prob=c(1/3,1/3,1/3), replace=TRUE)
  # print(totalStep_values)
  # generate the data:
  NPop = list()
  R0_vals = list()

  for (im_index in 1:clusters){
    # print(random_R0)
    if(random_R0){
    R0 <- runif(1, min=1.5, max=5)
    }
    # print(R0)
    tmp = generate_pd_matrix_HIV(NSamples=NSamples, R0=R0, totalStep=totalStep_values[im_index], spike_root = TRUE)
    Matrices[im_index,,] = tmp$Matrices
    NPop = append(NPop, tmp$NPop)
    R0_vals = append(R0_vals, R0)
  }
  true_label_vec = vector(length=NSamples*clusters)
  NPop_vec = vector(length=NSamples*clusters)
  R0_vec = vector(length=NSamples*clusters)
  root_position = NSamples + 1 # pre-compute for efficiency
  total_samples = (clusters * NSamples)
  megatrix = matrix(nrow=total_samples, ncol = total_samples)
  # unpack the main diagonal
  for (im_index in 0:(clusters-1)){
    data_in = matrix(Matrices[im_index+1,1:NSamples, 1:NSamples], NSamples, NSamples)
    megatrix[(1+im_index*NSamples):((im_index+1)*NSamples), (1+im_index*NSamples):((im_index+1)*NSamples)] = data_in[,]
    true_label_vec[(1+im_index*NSamples):((im_index+1)*NSamples)] <- rep(totalStep_values[im_index+1], NSamples)
    NPop_vec[(1+im_index*NSamples):((im_index+1)*NSamples)] <- rep(NPop[[im_index+1]][1], NSamples)
    R0_vec[(1+im_index*NSamples):((im_index+1)*NSamples)] <- rep(R0_vals[[im_index+1]][1], NSamples)
  }
  # sample initial  "star topology" distances
  star_distances = sample(1:3, clusters, replace=TRUE)
  # star_distances[i] is initial distance length for branch/sim `i`
  
  # this giant mess of for loops enables C-style array striding
  for (first_im_index in 1:(clusters)){
    for(first_sample in 1:NSamples){
      for(second_im_index in first_im_index:clusters){
        for (second_sample in 1:NSamples) {
          value = as.integer(star_distances[first_im_index] + star_distances[second_im_index]
                  + as.integer(Matrices[first_im_index, first_sample, root_position])
                  + as.integer(Matrices[second_im_index, second_sample, root_position]))
          mgxy = second_sample + (second_im_index-1) * NSamples  # megatrix y
          mgxx = first_sample + (first_im_index-1) * NSamples
          if(is.na(megatrix[mgxx, mgxy])){
            megatrix[mgxx, mgxy] = value
            megatrix[mgxy, mgxx] = value
          }
        }
        }
      }
  }
  if(shuffle){
    shuffle_order = sample(c(1:(NSamples*clusters)))
  } else {
    shuffle_order = c(1:(NSamples*clusters))
  }
  
  #print(shuffle_order)
  #print(true_label_vec)
  true_label_vec = true_label_vec[shuffle_order]
  NPop_vec = NPop_vec[shuffle_order]
  R0_vec = R0_vec[shuffle_order]
  megatrix[,] = megatrix[shuffle_order,]
  megatrix[,] = megatrix[,shuffle_order]
  return_list = list("matrix"=megatrix, "true_labels"=true_label_vec, "shuffle_order"=shuffle_order, "NPop"=NPop_vec, "R0"=R0_vec)
  return(return_list)
}

#bigloop_HIV(NSamples = 10, R0 = 10, totalStep = 10, itermax = 100)
bigloop_HIV <- function(NSamples,
                       R0,
                       totalStep, 
                       itermax) {
  biglist <- list(NULL)
  library(foreach)
  library(doParallel)
  n <- detectCores()
  if (n <= 16) {  # set aside a thread for desktop usage
    n <- n - 1
  }  # be careful here
  myCluster <- makeCluster(n, type = "PSOCK") # type of cluster
  registerDoParallel(myCluster)
  trials <- seq(1, itermax)
  biglist <-
    foreach (
      iterobj = trials,
      .export = c(
        "gen_Geneology_exp",
        "Pairwise_diff_HIV",
        "gen_Geneologies",
        "gen_trees_matrices",
        "generate_pd_matrix_HIV"
      )
    ) %dopar%
    {
      #print(paste('iterobj count: ', toString(iterobj), sep = ' '))
      generate_pd_matrix_HIV(
        NSamples = NSamples,
        R0 = R0,
        totalStep = totalStep,
        spike_root=FALSE
      )
    }
  stopCluster(myCluster)
  return(rebuild_named_list(biglist, "Matrices", "NPop"))
  
}

bigloop_HIV_mega <- function(NSamples,
                       R0,
                       totalStep, 
                       itermax, clusters) {
  biglist <- list(NULL)
  library(foreach)
  library(doParallel)
  n <- detectCores()
  if (n <= 16) {
    n <- n - 1
  }  # be careful here
  myCluster <- makeCluster(n, type = "PSOCK") # type of cluster
  registerDoParallel(myCluster)
  trials <- seq(1, itermax)
  biglist <-
    foreach (
      iterobj = trials,
      .export = c(
        "gen_Geneology_exp",
        "Pairwise_diff_HIV",
        "gen_Geneologies",
        "gen_trees_matrices",
        "generate_pd_matrix_HIV",
        "generate_mega_pd_matrix_HIV"
      )
    ) %dopar%
    {
      #print(paste('iterobj count: ', toString(iterobj), sep = ' '))
      generate_mega_pd_matrix_HIV(
        NSamples = NSamples,
        R0 = R0,
        totalStep = totalStep,
        spike_root=TRUE,
        clusters = 3
      )
    }
  stopCluster(myCluster)
  return(rebuild_named_list(biglist, "Matrices", "NPop"))
  
}

perf_check_HIV <- function(x = NULL) {
  library(profvis)
  profvis({
    for (ii in 1:1000)
      generate_pd_matrix_HIV(
        NSamples = 10,
        R0 = 10,
        totalStep = 10
      )
  })
}

rebuild_named_list <- function(inlist, ...){
  # Convert an ordered list accessed inlist[[i]]$label to inlist$label[[i]]
  # Assumes heterogeneity, attributes of each list index must be the same
  # as the first element.  
  
  # First get the 
  names_vector = names(inlist[[1]])
  newlist = vector(mode="list", length=length(list(...)))
  names(newlist) <- names_vector
  for(index in 1:length(list(...))){
    newlist[[names_vector[[index]]]] <- abind(lapply(inlist, access_, names=names_vector, index=index), along=1)
  }
  return(newlist)
}

access_ <- function(item, names, index){
  # Utility function to functionalize rebuild_named_list
  obj = item[[names[[index]]]]
  return(obj)
}
