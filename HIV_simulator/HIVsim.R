library(abind)
library(R.matlab)

#prob_birth=R0/24 #prob_death*R0
gen_Geneology_exp <- function(NSamples, R0, NPop, totalSteps, spike_root=FALSE) {
  #prob_death=0.05


  N = NPop


  Parents = matrix(0, max(totalSteps, 12) * N, 2)

  pp = 2
  currStep = 1
  active = c(1)
  birthStep = c(1)
  #endStep = rexp(rate=1/24,n=1)  # exponential model
  endStep = sample(13:36, 1) # for uniform model
  flag = TRUE
  startstep = 1000
  while (currStep < totalSteps + startstep) {

    indsRem = NULL

    L_active = length(active)
    if (flag && L_active > 0.9 * N && L_active > NSamples) {
      # print(c(L_active, NSamples))
      flag = FALSE
      startstep = currStep
    }


    L_active = length(active)
    if (N - L_active > 0) {
      offsprings = ((currStep - birthStep) < 3) * 0.4 / 3 * R0 / 0.505 + ((currStep - birthStep) >= 3) * 0.005 * R0 / 0.505
      tot_offsprings = ceiling(min(N - L_active, sum(offsprings)))
      probs = offsprings / sum(offsprings)

      prts = sample(1:L_active, tot_offsprings, replace = TRUE)
      Parents[pp:(pp + tot_offsprings - 1), 1] = active[prts]
      Parents[pp:(pp + tot_offsprings - 1), 2] = currStep
      active = c(active, pp:(pp + tot_offsprings - 1))
      birthStep = c(birthStep, rep(currStep + 1, tot_offsprings))
      endStep = c(endStep, currStep +
                    #rexp(rate=1/24, n=tot_offsprings)  # Exponential with mean 24 = 2 years
                    sample(13:36, tot_offsprings, replace =TRUE)  # uniform with mean 2 years
                  )
    }
    # print(c(currStep, endStep))
    indsRem = which(endStep <= currStep)  # remove anything we should
    len_Rem = length(indsRem)
    if (len_Rem > 0 && len_Rem < L_active) {  # don't remove the final infection
      active = active[-indsRem]
      endStep = endStep[-indsRem]
      birthStep = birthStep[-indsRem]
    }

    currStep = currStep + 1
    pp = pp + tot_offsprings
  }
  # We need to have set NSamples by here
  if (NSamples < 0){
    # If Nsamples is a negative number, we set the number of samples
    # as -Nsamples times the current population size.
    print("Trying to set the number of samples dynamically")
    # Magic key to return all
    NSamples=length(active) * (-1) * NSamples
    NSamples <- floor(NSamples)  # enforce type safety
    print(c(NSamples, length(active), 0 %in% active))
  }

  Samples = sample(active, NSamples, replace = FALSE)
  if(spike_root){
    Samples <- c(Samples, 0)  # add the root in here
    NSamples <- NSamples + 1  # increase the number of samples
  }

  Geneology <- matrix(0, 2 * NSamples, 5)
  Geneology[1:(2 * NSamples - 1), 1] <- c(1:(2 * NSamples - 1))

  # Taking into account of the sampling time
  Geneology[1:NSamples, 3] = currStep + rpois(NSamples, 6)
  if(spike_root){
    # Sample date for the initial infection is 0, no tip adjustment
    Geneology[NSamples,3] <- 0  # starts at the origin
  }
  nNodes = NSamples
  IDs = 1:NSamples
  pp = NSamples + 1

  while (nNodes > 1) {
    ind = which.max(Parents[Samples, 2])
    parent = Parents[Samples[ind], 1]
    coall = which(Samples == parent)
    if (length(coall) == 0) {
      Samples[ind] = parent
    } else {
      Geneology[pp, 3] = Parents[Samples[ind], 2]

      Geneology[IDs[ind], 2] = pp
      Geneology[IDs[coall], 2] = pp
      if (!is.na(IDs[coall])) {  # Only happens when we try to backprop from the first infection
        if (Geneology[IDs[coall], 3] < Parents[Samples[ind], 2])
          Geneology[IDs[coall], 3] = Parents[Samples[ind], 2]
      }
      IDs[coall] = pp
      Samples = Samples[-ind]
      IDs = IDs[-ind]
      pp = pp + 1
      nNodes = nNodes - 1
    }
  }
  print('x')
  inds = 1:(2 * NSamples - 1)
  Geneology[inds, 3] = Geneology[inds, 3] - Geneology[(2 * NSamples - 1), 3]
  # Subtract off from the root
  Geneology[inds, 3] = max(Geneology[inds, 3]) - Geneology[inds, 3]
  # Happens to place the root back at 0 if we spiked in the origin


  inds = 1:(2 * NSamples - 2)
  Geneology[inds, 4] = Geneology[Geneology[inds, 2], 3] - Geneology[inds, 3]
  Geneology[inds, 5] = Geneology[inds, 4] / (sum(Geneology[inds, 4]))
  return(Geneology)
}

gen_Geneologies <- function(NSamples, R0, NPop, totalSteps, spike_root=FALSE) {
  trees = list()
  Geneology = gen_Geneology_exp(NSamples, R0, NPop, totalSteps, spike_root=spike_root)
  trees[[1]] = Geneology
  return(trees)
}

Pairwise_diff_HIV <- function(Geneology, spike_root = FALSE) {
  # Convert a geneology to a pairwise difference matrix
  # spike_root option adds a row and column for a node at the root

  M = length(Geneology[, 1]) / 2

  # if (spike_root) {
  #   M <- M + 1
  # }  # 1 more root node for multicluster data
  mrca_mat <- matrix(0, M, M)
  pair_diff_mat <- matrix(0, M, M)
  ind_par_mat <- Geneology
  # if (spike_root) { M <- M - 1 }
  ## Obtain an MRCA matrix
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
  # if (spike_root) {
  #   oldest_node = length(Geneology[, 1]) - 1
  #   M <- M + 1
  #   mrca_mat[M,] = oldest_node
  #   mrca_mat[, M] = oldest_node
  #   mrca_mat[M, M] = 0  #
  # }
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
  return(pair_diff_mat)
}


gen_trees_matrices <- function(Geneologies, Nmuts, spike_root = FALSE) {
  if (!is.vector(Geneologies)) {
    Geneologies = list(Geneologies)
  }
  NGeneologies = length(Geneologies)
  NSamples = length(Geneologies[[1]][, 1]) / 2

  Trees = list()
  # This uses non-type-safe addition: int + bool = int
  Matrices = array(NA, dim = c(1, NSamples, NSamples))

  id = 1
  for (ii in 1:NGeneologies) {
    if (!is.null(Geneologies[[ii]])) {
      Geneology = Geneologies[[ii]]
      inds = 1:(2 * NSamples - 2)
      Geneology[inds, 5] = rpois(length(inds), Geneology[inds, 4] * Nmuts)
      # taking lambda=0 here is not a problem - R describes it as a point mass at 0
      res = Pairwise_diff_HIV(Geneology, spike_root = spike_root)
      Matrices[id, ,] = res

      Trees[[id]] = Geneology

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
  # Sample code for testing the simulator
  NSamples = 20
  NumRep = 3

  R0s = c(5, 15)  # test a few different R0 values
  TotalSteps = 12 * c(0, 1, 2, 10)  # time steps

  print(length(R0s) * length(TotalSteps))
  TrainingSet = list(Matrices = NULL, labels = NULL)

  par(mfrow = c(2, 4), mar = c(2, 2, 0.1, 0.1))

  for (jj in 1:length(R0s)) {
    R0 = R0s[jj]
    for (kk in 1:length(TotalSteps)) {
      totalStep = TotalSteps[kk]
      print(c(R0, totalStep))

      TREEs = list(NULL)
      tp = 1
      for (ii in 1:NumRep) {
        Nmut = 0.0067 * 300 / 12  # don't change me  # use value from Leitner et al. 1999
        NPop = floor(10^runif(1, min = 3, max = 4)) # rand uniform, NOT run if
        Geneologies = gen_Geneologies(NSamples, R0, NPop, totalStep)
        out = gen_trees_matrices(Geneologies, Nmut)
        TrainingSet$Matrices = abind(TrainingSet$Matrices, out$Matrices, along = 1)
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
      sorted_hhc = reorder(hhc, dd, method = "OLO")
      verbose = FALSE
      if (verbose == TRUE) {
        print(hhc$order)
        plot(hhc)
        print(sorted_hhc$order)
        plot(sorted_hhc)
        col = rev(heat.colors(999))
        stats::heatmap(out$Matrices[1, sorted_hhc$order, sorted_hhc$order], symm = TRUE, Rowv = NA, Colv = "Rowv", col = col)
        stats::heatmap(out$Matrices[1, hhc$order, hhc$order], symm = TRUE, Rowv = NA, Colv = "Rowv", col = col)
      }
      out$Matrices = out$Matrices[, sorted_hhc$order, sorted_hhc$order, drop = FALSE]
      print(stats::heatmap(out$Matrices[, ,], symm = TRUE, Rowv = NA, Colv = "Rowv"))

      return(out$Matrices)
    }
  }

  print(dim(TrainingSet$Matrices))
  filenameMat = 'HIV_Train/TrainingSet_Exp_20.mat'
  writeMat(filenameMat,
           matrices = TrainingSet$Matrices,
           labels = TrainingSet$labels)
}

generate_pd_matrix_HIV <- function(NSamples = 20,
                                   R0,
                                   totalStep,
                                   spike_root = FALSE) {
  # spike_root adds an extra row/column to the evolutionary pairwise distance matrix
  # for the distance from the root node.
  Nmut = 0.006 * 300 / 12 #don't change me  - set at 0.002
  NPop = floor(10^runif(1, min = 3, max = 4))

  R0 = runif(1, 1.5, 5)  # override R0 value passed in

  Geneologies = gen_Geneologies(NSamples, R0, NPop, 12 * totalStep, spike_root=spike_root) #here's the workhorse
  out = gen_trees_matrices(Geneologies, Nmut, spike_root = spike_root)
  out = list("Matrices" = out$Matrices, "NPop" = NPop, "R0" = R0)
  return(out) # $Matrices
}

# tmp = generate_mega_pd_matrix_HIV(NSamples = cluster_sample_size, R0 = R0_default, random_R0 = randomize_R0_vals, clusters=number_of_clusters, shuffle=shuffle_in_cluster)


reduce_bigmat <- function(oversampled_matrix, N_expected, density_level, root_position){
  # N_expected - how many sequences to consider. Do not include the root infection
  # density_level - float between 0 and 1. How much of the population to subsample.
  # oversampled_matrix - matrix of full population

  # Flatten down to a matrix
  # oversampled_matrix <- adrop(oversampled_matrix, drop=1)
  Ndata <- dim(oversampled_matrix)[2]  # how many sequences in the dataset, [# of mats, nrows, ncols]
  # sample_size <- min(1, ceiling(density_level * Ndata))
  sample_size <- N_expected
  mask <- rep(TRUE, Ndata)
  mask[Ndata] <- FALSE  # Location of root is hard coded here

  sample <- sample(x=Ndata, size=1, replace=FALSE, prob=mask/sum(mask))
  # which row/col to go to
  # neighbors <- sample(x=nrow(oversampled_matrix), size=N, replace=FALSE, prob=mask/sum(mask))
  # subtract 1 to not interfere with the spiked in root
  print(oversampled_matrix[, sample, ])
  new_order <- sort(oversampled_matrix[, sample, ], index.return=TRUE)
  closest_k <- new_order$ix[1:N_expected]
  print(new_order)
  closest_k <- as.vector(append(closest_k, Ndata))  # add back in the last value, we need it
  print(closest_k)
  # Now subset the matrix
  data_matrix <- oversampled_matrix[, closest_k, closest_k]
  print(dim(data_matrix))
  # data_matrix <- data_matrix[, , closest_k]

  return(data_matrix)

}

generate_mega_pd_matrix_HIV <- function(NSamples,
                                        R0, random_R0 = TRUE, density_subsampler=FALSE, density_level=1,
                                        clusters = 3, shuffle = TRUE) {
  spike_root = TRUE
  Matrices = array(NA, dim = c(clusters, NSamples + spike_root, NSamples + spike_root))
  # One option here is to take equal amounts of each data
  #totalStep_values = sample(x = c(0, 2, 10), size = clusters,
  #                          prob = c(1 / 3, 1 / 3, 1 / 3), replace = TRUE)
  nzeros = clusters %/% 2  # integer division
  totalStep_values = c(rep(0, nzeros), sample(x=c(2,10), size=clusters-nzeros, prob=c(1/2, 1/2), replace=TRUE))
  print(totalStep_values)
  # generate the data:
  NPop = list()
  R0_vals = list()

  for (im_index in 1:clusters) {
    # print(random_R0)
    if (random_R0) {
      R0 <- runif(1, min = 1.5, max = 5)
    }
    # print(R0)
    if(density_subsampler){
      NSamples_safe <- NSamples  # store me for later
      NSamples <- -1  # status code to indicate "sample all"
    }

  root_position = NSamples + 1 # pre-compute for efficiency
  if(density_subsampler){
    sample_size <- -1 * density_level
  }
  else{
    sample_size <- NSamples
  }
    tmp = generate_pd_matrix_HIV(NSamples = sample_size, R0 = R0,
                                 totalStep = totalStep_values[im_index], spike_root = TRUE)
    if(density_subsampler){
      reduced_matrix = reduce_bigmat(oversampled_matrix=tmp, N_expected=NSamples_safe, root_loc = root_position)
      Nsamples = NSamples_safe  # put the value back
      tmp$Matrices = reduced_matrix
    }
    Matrices[im_index, ,] = tmp$Matrices
    NPop = append(NPop, tmp$NPop)
    R0_vals = append(R0_vals, R0)
  }
  true_label_vec = vector(length = NSamples * clusters)
  NPop_vec = vector(length = NSamples * clusters)
  R0_vec = vector(length = NSamples * clusters)
  total_samples = (clusters * NSamples)
  megatrix = matrix(nrow = total_samples, ncol = total_samples)

  # unpack the main diagonal
  for (im_index in 0:(clusters - 1)) {
    data_in = matrix(Matrices[im_index + 1, 1:NSamples, 1:NSamples], NSamples, NSamples)
    megatrix[(1 + im_index * NSamples):((im_index + 1) * NSamples), (1 + im_index * NSamples):((im_index + 1) * NSamples)] = data_in[,]
    true_label_vec[(1 + im_index * NSamples):((im_index + 1) * NSamples)] <- rep(totalStep_values[im_index + 1], NSamples)
    NPop_vec[(1 + im_index * NSamples):((im_index + 1) * NSamples)] <- rep(NPop[[im_index + 1]][1], NSamples)
    R0_vec[(1 + im_index * NSamples):((im_index + 1) * NSamples)] <- rep(R0_vals[[im_index + 1]][1], NSamples)
  }
  # sample initial  "star topology" styled distances
  star_distances = rep(15,  clusters)  # + sample(1:3, clusters, replace = TRUE)  # 10% of 300 = 30, place roots farther apart
  # star_distances[i] is initial distance length for branch/sim `i`

  # this giant mess of for loops enables C-style array striding
  for (first_im_index in 1:(clusters)) {
    for (first_sample in 1:NSamples) {
      for (second_im_index in first_im_index:clusters) {
        for (second_sample in 1:NSamples) {
          value = as.integer(star_distances[first_im_index] +
                               star_distances[second_im_index]
                               +
                               as.integer(Matrices[first_im_index, first_sample, root_position])
                               +
                               as.integer(Matrices[second_im_index, second_sample, root_position]))
          mgxy = second_sample + (second_im_index - 1) * NSamples  # megatrix y
          mgxx = first_sample + (first_im_index - 1) * NSamples
          if (is.na(megatrix[mgxx, mgxy])) {
            megatrix[mgxx, mgxy] = value
            megatrix[mgxy, mgxx] = value
          }
        }
      }
    }
  }
  if (shuffle) {
    shuffle_order = sample(c(1:(NSamples * clusters)))
  } else {
    shuffle_order = c(1:(NSamples * clusters))
  }

  #print(shuffle_order)
  #print(true_label_vec)
  true_label_vec = true_label_vec[shuffle_order]
  NPop_vec = NPop_vec[shuffle_order]
  R0_vec = R0_vec[shuffle_order]
  megatrix[,] = megatrix[shuffle_order,]
  megatrix[,] = megatrix[, shuffle_order]
  return_list = list("matrix" = megatrix, "true_labels" = true_label_vec, "shuffle_order" = shuffle_order, "NPop" = NPop_vec, "R0" = R0_vec)
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
    foreach(
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
        spike_root = FALSE
      )
    }
  stopCluster(myCluster)
  return(rebuild_named_list(biglist, "Matrices", "NPop", "R0"))

}

bigloop_HIV_mega <- function(NSamples,
                             R0,
                             totalStep,
                             itermax,
                             clusters,
                             density_subsampler=FALSE,
                             density_level=1) {
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
    foreach(
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
      # random_R0 = TRUE, density_subsampler=FALSE, density_level=1,
      #                                   clusters = 3, shuffle = TRUE
      generate_mega_pd_matrix_HIV(
        NSamples = NSamples,
        R0 = R0, density_subsampler = density_subsampler,
        density_level=density_level,
        # totalStep = totalStep,
        # spike_root = TRUE,
        clusters = clusters
      )
    }
  stopCluster(myCluster)
  return(rebuild_named_list(biglist, "matrix", "true_labels", "shuffle_order", "NPop", "R0"))

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

rebuild_named_list <- function(inlist, ...) {
  # Convert an ordered list accessed inlist[[i]]$label to inlist$label[[i]]
  # Assumes heterogeneity, attributes of each list index must be the same
  # as the first element.

  # First get the
  names_vector = names(inlist[[1]])
  newlist = vector(mode = "list", length = length(list(...)))
  names(newlist) <- names_vector
  for (index in 1:length(list(...))) {
    newlist[[names_vector[[index]]]] <- abind(lapply(inlist, access_, names = names_vector, index = index), along = -1)
  }
  return(newlist)
}

access_ <- function(item, names, index) {
  # Utility function to functionalize rebuild_named_list
  obj = item[[names[[index]]]]
  return(obj)
}
