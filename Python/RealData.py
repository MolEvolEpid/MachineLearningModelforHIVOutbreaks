"""
RealData.py
Michael Kupperman

Handle data as if it was real.

"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hclust
import scipy.io as scio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import squareform

from typing import Union


class RealData(object):
    def __init__(self, source, source_type, scale_factor=1):
        """ Import a dataset styled after real data. Not all fields will be populated.
        Args:
            source_type: One of 'real', 'synthetic', or '_dict'.
        keyword `source_type` indicates which values to unpack from `source_type`.
        Source type '_dict' should only be used for """
        # Set of key values loaded - check for overlap before extend
        self.loaded = set()

        # Always added
        self.matrices = None

        # only added for real data
        self.row_labels = None

        # Only added for synthetic data
        self.labels = None
        self.shuffle = None
        self.NPop = None
        self.times = None
        self.num_clusters = None
        self.size_clusters = None
        self.R0 = None
        self.cluster_size = None

        self.source_type = source_type
        if source_type == 'real':
            self.load_real_data(source, scale_factor=scale_factor)
        elif source_type == 'synthetic' or source_type == 'synth':
            self.load_synth_data(source)
            self.source_type = 'synth'
        elif source_type == '_dict':
            self.load_from_dict(source)
        else:
            raise NotImplementedError(f'The source type {source_type} is not implemented')

    def load_real_data(self, source_file, scale_factor):
        """Real data offers limited info, we only want to load in the known fields.


        Args:
            source_file: String path to data `.mat` file.
            scale_factor: rescale matrix by factor, useful for seq length/evolutionary distance conversions

        Returns:
            None

        """

        data = scio.loadmat(source_file)
        self.matrices = data['matrices']
        self.loaded.add('matrices')
        self.matrices = self.matrices * scale_factor
        self.row_labels = data['row_labels']
        self.loaded.add('row_labels')

    def load_synth_data(self, source_file):
        """
        synthetic data offers generation metadata which we want to store.
        From R:
          clusters=number_of_clusters,
          cluster_size=cluster_sample_size,
          matrices = data,    # the large composite matrices, in a 3d array
          labels = labels,    # labels for each person
          shuffle = shuffle,  # store the shuffle maps
          NPop= NPop          # store the simulation pop size for each person
          Some data objects may come with an additional R0 attribute.
        """

        data = scio.loadmat(source_file)

        self.matrices = data['matrices']
        self.loaded.add('matrices')
        self.labels = data['labels']
        self.loaded.add('labels')
        if 'shuffle' in data:
            self.shuffle = data['shuffle']
            self.loaded.add('shuffle')
        self.NPop = data['NPop']
        self.loaded.add('NPop')
        if 'cluster_size' in data:
            self.cluster_size = data['cluster_size']
            self.loaded.add('cluster_size')
        if 'clusters' in data:
            self.num_clusters = data['clusters']
            self.loaded.add('clusters')
        if 'times' in data:
            self.times = data['times']
            self.loaded.add('times')
        print(data.keys())
        if 'R0' in data.keys():
            print('Adding R0')
            self.R0 = data['R0']
            self.loaded.add('R0')
        elif 'values' in data:
            self.R0 = data['values'][:, 0]
            self.loaded.add('values')

    def load_from_dict(self, data):
        """ Load data from a dictionary. Useful for generating subsets from larger sets. """
        self.matrices = data['matrices']
        self.loaded.add('matrices')
        self.row_labels = data['row_labels']
        self.loaded.add('row_labels')
        self.source_type = 'real'  # We got years, so real data
        self.loaded.add('source_type')

    def extend(self, other):
        """ Extend self by another Realdata object with the same defined attributes. """
        assert type(other) is type(self), 'cannot extend by a different type'
        assert self.loaded == other.loaded, 'Both objects must have the same data fields initialized'
        for field in self.loaded:
            # Get data
            attr1 = getattr(self, field)
            attr2 = getattr(other, field)
            # concatenate
            attr12 = np.concatenate((attr1, attr2), axis=0)
            # store
            setattr(self, field, attr12)

    def _real_to_dict(self):
        """ Return a copy of the matrices and row labels.

        See `load_from_dict` method above.
        """
        row_labels = self.row_labels.copy() if self.row_labels is not None else None
        return {'matrices': self.matrices.copy(), 'row_labels': row_labels}

    def __len__(self):
        if self.matrices is not None:
            return self.matrices.shape[0]
        else:  # we don't have any data, length is zero
            return 0

    def predict_by_year(self, model, window_size, choice_method, rule='forwards', matrix=1,
                        cluster_method='None', second_clustering_method='None', return_map=False):
        """ Compute predictions on data by year using model.predict() method.
        Returns dictionaries for predictions, year-subset source data, and sort aggregates
        """

        data_subset, sort_dict = self.subset_by_year(min_size=window_size, rule=rule)
        pred_dict = {}
        rd_dict = {}
        for key in data_subset.keys():
            rd_tmp = RealData(source=data_subset[key], source_type='_dict')
            rd_dict[key] = rd_tmp
            preds = rd_tmp.predict(model=model, window_size=window_size, choice_method=choice_method,
                                   matrix=matrix, cluster_method=cluster_method,
                                   second_clustering_method=second_clustering_method, return_map=return_map)
            pred_dict[key] = preds
        return pred_dict, rd_dict, sort_dict

    def subset_by_year(self, min_size=15, rule='forwards'):
        """ generate a dict of RealData with infections sorted by year. 
        Only return images with more than `min_size` elements. Specify
        `rule=forward` to join samples that are less than min_size
        with the next year's sample. see `join_dict_by_size` for details on the joining.
        """
        assert type(min_size) is int
        assert min_size > 1, 'An image cannot have less than 1 infection'
        year_dict = self._filter_by_year()
        data = self._real_to_dict()
        new_subset = {}
        # Filter
        if rule == 'forwards':
            year_dict, sort_dict = join_dict_by_size(min_size=min_size, data=year_dict, rule='forwards')
        elif rule == 'backwards':
            year_dict, sort_dict = join_dict_by_size(min_size=min_size, data=year_dict, rule='backwards')
        elif rule == 'None' or type(rule) is None:
            sort_dict = {key: [key] for key in year_dict.keys()}
        else:
            raise ValueError(f'Rule {rule} was not a recognized option')
        # Build images
        for year in year_dict.keys():  # loop over each year
            indiv = np.asarray(year_dict[year])  # sort indexes and cast to numpy array
            num_indiv = indiv.shape
            new_im = data['matrices'].copy()  # fresh copy
            # Subset by row and by column, one at a time
            new_im = new_im[indiv, :]
            new_im = new_im[:, indiv]
            # print(new_im)
            new_labels = data['row_labels'][indiv]
            new_subset[year] = {'matrices': new_im, 'row_labels': new_labels}
        return new_subset, sort_dict

    def _filter_by_year(self):
        """ Generate a dict of infections by year.

        A utility method for filtering the data by year. Enables predictions based on each year, if sample order is
        not known.

        """
        assert self.source_type == 'real', 'Cannot filter by year without labels present'
        years = []
        # Build a list of sample years
        for label in self.row_labels.ravel():
            tokens = np.char.split(label, sep='.').item()
            year_set = False
            # noinspection PyTypeChecker
            for token in tokens:
                try:
                    if int(token) > 2100 or int(token) < 1980:
                        raise ValueError  # break the try
                    year = int(token)

                    year_set = True
                except ValueError:
                    pass  # do nothing
                if not year_set:
                    year = 0  # assign it to year 0,
            # noinspection PyUnboundLocalVariable
            years.append(year)
        # Generate a dict to store results
        samples_per_year = {key: [] for key in years}
        for idx in range(len(years)):
            # make a list of infection indices by year
            samples_per_year[years[idx]].append(idx)
        for key in samples_per_year.keys():
            samples_per_year[key].sort()
        return samples_per_year

    def _visualize_matrix(self, index, cluster_method='None', preds=None, gt=None,
                          fig=None, ax=None, create_cbar=False):
        """ Plot the matrix.

        Options support clustering & prediction overlays, ground truth values, and to create a new
        colorbar axis or use the default constructor.
        """

        if ax is None and fig is None:
            fig, ax = plt.subplots(1)
        image = self._get_image(index, cluster_method=cluster_method, return_map=False)
        # image = np.squeeze(image[0])
        vmax = np.max(image)
        image = np.squeeze(image)
        main_IM = ax.imshow(image, cmap='plasma', vmin=0, vmax=vmax, interpolation='nearest')
        if create_cbar:
            # Create the 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            plt.colorbar(main_IM, cax=cax)
        else:
            fig.colorbar(main_IM, ax=ax)
        if preds is not None:
            overlay_preds(axes=ax, preds=preds, edge_color='y', face_color='y', ls=None)
        if gt is not None:
            print('Adding ground truth values to overlay')
            overlay_preds(axes=ax, preds=gt, edge_color='r', face_color='r', ls='--', alpha=0.4)
        return fig, ax

    def show(self, index, cluster_method='None', second_clust_method='None', model=None, fig_obj=None,
             create_cbar=False, window_size=None,
             overlay_preds=False, highlight=0, choice_method='argmax', ax=None,
             add_gt=False):
        """
        Return a pyplot figure and axes for matrices[index].

        Args:
            index:
            fig_obj:
            second_clust_method:
            choice_method:
            create_cbar:
            cluster_method: string, method for first clustering.
            model: An object with a predict method. See predict method for this class for more information
            window_size: integer, see class predict method for more information
            overlay_preds: bool, add boxes to plot where pred=highlight
            highlight: int, Model label to highlight.
            ax (pyplot axis): Optionally attach the plot to a specified axis.
            add_gt: bool, add ground-truth labels to predictive overlay.
        """

        gt = None
        preds = None
        if overlay_preds:
            assert model is not None, 'A model must be provided'
            assert highlight is not None and type(highlight) is int, 'highlight must be an int'
            assert window_size is not None, 'Window size must be specified'
            preds, idx_map = self.predict(model=model, window_size=window_size, cluster_method=cluster_method,
                                          matrix=index,
                                          second_clustering_method=second_clust_method, choice_method=choice_method,
                                          return_map=True)
            preds = preds[idx_map]  # We want the sorted predictions, not the initial inputs
            preds = np.equal(preds, highlight)  # only pass the logical indices where we want boxes
            if add_gt:
                # Compute the correct overlay based on known labels.
                # Will fail if source_type='real' at data import
                gt = self.labels[index][idx_map] == (highlight + 1)
                print('gt is:', gt)
        return self._visualize_matrix(index, cluster_method=cluster_method, preds=preds, ax=ax, gt=gt,
                                      fig=fig_obj, create_cbar=create_cbar)

    def lineplot(self, index: int, models: list = None, window_sizes: list[Union[int, list]] = None,
                 choice_method: str = 'argmax', cluster_method: str = 'HC', second_cluster_method: str = 'None',
                 cmap_key: str = 'Set2', show_mat: bool = True):
        """ Generate a 1d line plot showing "alignment"-style predictions.
        """
        if window_sizes is None:
            window_sizes = []
        if models is None:
            models = []

        preds_list = []
        if type(models) is not list:
            models = [models]
        if type(window_sizes) is not list:
            window_sizes = [window_sizes]
        for model, window_size in zip(models, window_sizes):
            preds, idx_map = self.predict(model=model, window_size=window_size, cluster_method=cluster_method,
                                          matrix=index,
                                          second_clustering_method=second_cluster_method, choice_method=choice_method,
                                          return_map=True)
            preds = preds[idx_map]  # We want the sorted predictions, not the initial inputs
            preds_list.append(preds)
        gt = None
        if self.source_type == 'synth':
            gt = self.labels[index, idx_map] - 1

        fig, ax, heatmap = map_alignment(gt=gt, preds=preds_list, cmap_key=cmap_key, add_second_axis=show_mat)
        fig.suptitle(f'Image {index} with {self.matrices.shape[1]} Infections')
        return fig, ax

    def _get_image(self, index, cluster_method='None', return_map=False):
        """
        Return a tuple of the matrix specified by index coersed
        into the correct shape and apply the specified cluster_method
        and also returns the ordering map computed. 
        """

        dim = len(self.matrices.shape)
        # We need to copy here to avoid side effects
        if dim == 2:
            matrix = np.expand_dims(self.matrices.copy(), 2)  # image is 2d, flat
        else:
            matrix = np.expand_dims(self.matrices[index, :, :].copy(), 2)

        idx, matrix = self._sort(index=index, matrix=matrix, cluster_method=cluster_method)

        if return_map:
            return matrix, idx
        else:
            return matrix

    @staticmethod
    def _sort(index, matrix, cluster_method):
        """ A wrapper to determine the sorting method that should be applied. Sort is an external function """
        if cluster_method == 'None':
            idx = np.asarray([index for index in range(matrix.shape[1])])
        elif cluster_method == 'OLO':
            matrix, idx = leaf_order_optimize(matrix)
        elif cluster_method == 'HC':
            matrix, idx = hc(matrix)
        else:
            raise NotImplementedError(f'Cluster method {cluster_method} was not recoginzed')
        return idx, matrix

    @staticmethod
    def batch_sort(image, method):
        """ Apply sort to each layer of image[idx,:,:] """
        image_new = np.empty(shape=image.shape)
        orderings = np.empty(shape=(image.shape[0], image.shape[1]), dtype=np.int)
        # switch on method
        if method == 'None':
            method = no_sort
        elif method == 'OLO':
            method = leaf_order_optimize
        elif method == 'HC':
            method = hc
        else:
            raise ValueError('method not recognized')

        # Apply the method to each layer
        for layer_id in range(image.shape[0]):
            im_new, order = method(image[layer_id, :, :, 0])
            im_new = im_new[:, :, np.newaxis]
            image_new[layer_id, :, :, :] = im_new
            orderings[layer_id, :] = order

        return image_new, orderings

    def get_ordered_labels(self, index, cluster_method):
        image_full, idx_map = self._get_image(index=index, cluster_method=cluster_method, return_map=True)
        labels = self.labels[index, :]
        labels = labels[idx_map]
        return labels

    def predict(self, model, window_size, choice_method, matrix=1, cluster_method='None',
                second_clustering_method='None', return_map=False, return_labels=False):
        """
        Evaluate the dataset with the `predict` method of input model.
        Predict should return the predicted label, not a probability distribution
        window_size is specific to the model.
        Argument choice_method specified the method to assign final labels for each
        individual. 
        Argument matrix specifies the index of the matrix in data.
        return: np.ndarray of predicted label
        """

        num_sub_images = self.matrices.shape[1] - window_size + 1
        scores = [[] for _ in range(self.matrices.shape[1])]  # per person scores
        image_full, idx_map = self._get_image(index=matrix, cluster_method=cluster_method, return_map=True)
        for index in range(0, num_sub_images, 128):  # step by batches
            # print(image_full.shape)
            batch_size = min(num_sub_images - index, 128)  # ensure last batch doesn't read off array
            image = stride_image_into_tensor(image_full, window_size=window_size, start=index, batch_size=batch_size)
            image, idx_map_second = self.batch_sort(image=image, method=second_clustering_method)
            result = model.predict(image)
            for im_idx in range(batch_size):
                # Loop over multiple images
                for person in range(window_size):
                    # use idx_map to correctly assign scores
                    scores[idx_map[im_idx + index + idx_map_second[im_idx, person]]].append(
                        result[im_idx].astype(np.int).item(0))
        person_predictions = self.compute_scores(scores, choice_method)
        if self.source_type == 'real':
            active = [self.row_labels[row_index] for row_index in range(len(self.row_labels)) if
                      person_predictions[row_index] == 0]
        if return_labels:
            # Make lists of 
            label_dict = {0: [], 1: [], 2: []}  # Our case only has 3
            for row_index in range(len(self.row_labels)):
                label_dict[person_predictions[row_index]].append(self.row_labels[row_index].item().item())
                # double item does the correct access through the array of arrays... yikes numpy

        if return_labels:
            if return_map:
                return person_predictions, idx_map, label_dict
            else:
                return person_predictions, label_dict
        else:
            if return_map:
                return person_predictions, idx_map
            else:
                return person_predictions

    def evaluate(self, model, window_size, choice_method, matrix=1, cluster_method='None'):
        """
        Evaluate the performance of a model and choice method on the dataset.
        This method should be called only on synthetic data when true labels are known.
        
        Returns the accuracy, predictions, and associated labels.
        """
        if type(matrix) is int:
            matrix = [matrix]
        acc_vals = list()
        pred_list = list()
        label_list = list()
        for matrix_id in matrix:
            predictions, idmap = self.predict(model=model, window_size=window_size,
                                              choice_method=choice_method, matrix=matrix_id,
                                              cluster_method=cluster_method, return_map=True)
            bool_filter = predictions != -1
            predictions_filtered = predictions[idmap]
            acc_vals.append(
                np.sum(np.equal(predictions_filtered, self.labels[matrix_id, idmap])) / np.sum(bool_filter))
            pred_list.append(predictions_filtered)
            label_list.append(self.labels[matrix_id, idmap])
        acc_np = np.array(acc_vals)
        print(
            f'Average score: {np.mean(acc_np)}\nmax score: {np.amax(acc_np)}\nmin score: {np.amin(acc_np)}\n SD: {np.std(acc_np)}\nmedian score: {np.median(acc_np)}')
        return acc_np, pred_list, label_list

    def compute_scores(self, scores, choice_method):
        """ A switch statement to match choice_method, scores are passed through"""
        if choice_method == 'inclusive' or choice_method == 'strict':
            return self._score_strict(scores)
        elif choice_method == 'argmax':
            return self._score_argmax(scores)
        elif choice_method == 'median':
            return self._score_median(scores)
        else:
            raise NotImplementedError(f'The choice method {choice_method} is not implemented')

    def _score_strict(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        return score_strict_(scores)

    def _score_argmax(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        return score_argmax_(scores)

    def _score_median(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        people_scores = np.zeros(shape=(len(scores)))
        for index, person_scores in enumerate(scores):
            person_scores = [value for value in scores[index] if value != None]
            x = np.rint(np.asarray(person_scores)).flatten().astype('int8')
            # print(self.labels[0, index] - 1, ' - ', x)
            # bc = np.bincount(x)
            people_scores[index] = np.median(x)
        return people_scores


def no_sort(matrix):
    """ A simple method to perform an identity transform."""
    order = np.array([val for val in range(matrix.shape[1])])
    return matrix, order


def leaf_order_optimize(matrix, method='ward'):
    distvec = squareform(matrix[:, :, 0])
    linkage_map = hclust.linkage(distvec, method=method)
    optimal_linkage_map = hclust.optimal_leaf_ordering(Z=linkage_map, y=distvec)
    order = hclust.leaves_list(optimal_linkage_map)
    matrix[:, :, 0] = matrix[order, :, 0]
    matrix[:, :, 0] = matrix[:, order, 0]
    return matrix, order


def hc(matrix, method='ward'):
    distvec = squareform(matrix[:, :, 0])
    linkage_map = hclust.linkage(distvec, method=method)
    order = hclust.leaves_list(linkage_map)
    matrix[:, :, 0] = matrix[order, :, 0]
    matrix[:, :, 0] = matrix[:, order, 0]
    return matrix, order


def stride_image_into_tensor(data, start, window_size, batch_size=32):
    """
    Stride window of k x k over d x d x 1 data into n x k x k x 1 tensor.
    """
    tensor = np.empty(shape=(batch_size, window_size, window_size, 1))
    for row in range(batch_size):
        idx = start + row
        end = idx + window_size
        tensor[row, :, :, :] = data[(idx):(end), (idx):(end), :]
    return tensor


def score_strict_(scores):
    people_scores = np.empty(shape=(len(scores)))
    for index, person_scores in enumerate(scores):
        x = np.rint(np.asarray(person_scores)).flatten().astype(np.int)
        people_scores[index] = np.amin(x)
    return people_scores


def score_argmax_(scores):
    people_scores = np.empty(shape=(len(scores)))
    for index in range(len(scores)):
        person_scores = scores[index]  # [value for value in scores[index] if value != None]
        x = np.rint(np.asarray(person_scores)).astype(np.int).flatten()
        bc = np.bincount(x)
        people_scores[index] = np.argmax(bc)
    return people_scores


def overlay_preds(axes, preds, edge_color='w', face_color='w', ls=None, alpha=0.7):
    """ Add tiles to axes where preds=True """
    # parse the list of preds to find the regions where true
    start = None
    end = None
    reset_flag = False
    len_patch = 0
    for idx in range(preds.shape[0]):
        if preds[idx]:  # add the point to the patch
            if start is None:
                start = idx
            len_patch += 1  # grow the patch
        else:  # preds is false and not at the end
            if start is not None:  # We have an open patch
                end = idx - 1  # last position was the last good position
                reset_flag = True
        if idx == (preds.shape[0] - 1):  # catch edge case, last value in matrix
            if start is not None:
                end = idx
                reset_flag = True

        if reset_flag:  # patch is finished, draw it and reset for the next patch
            # update plot
            rect = patches.Rectangle((start, start), (end - start), (end - start),
                                     linewidth=1, edgecolor=edge_color, facecolor=face_color, linestyle=ls, alpha=alpha)
            axes.add_patch(rect)
            print(start, end)
            # reset counters and flags
            start = None
            end = None
            len_patch = 0
            reset_flag = False


def is_np(array):
    """ Check that the type of the input is a numpy array in a functional pattern"""
    return type(array) is np.ndarray


def map_alignment(gt, preds, cmap_key='viridis', add_second_axis=False):
    """ plot an alignment map of preds against the ground truth `gt` 
    gt and rows of preds must be of the same length
    
    :param gt: array-like of ground truth prediction
    :param preds: list of array-like or array-like predictions    
    :param add_second_axis: bool, add a second axis on the right side. Allows for plotting source data elsewhere

    Args:
        cmap_key: color map key. Default is viridis.
        gt: Ground Truth labels
        add_second_axis (bool): Add a second axis to the figure.
    """
    gt_offset = 0
    if gt is not None:
        gt = np.asarray(gt)
        if type(preds) is list:
            # assert all(map(preds, is_np)), 'at least one pred list item is not a numpy array'
            assert all([ar.shape[0] == gt.shape[0] for ar in preds]), 'All arrays must be the same shape'
            preds = np.asarray(preds)
        elif type(preds) is np.ndarray:
            assert preds.shape[1] == gt.shape[0]  # Check the lengths are the same

        nrow = 1 + preds.shape[0]  # gt + samples
        ncol = gt.shape[0]  # alyready checked for same shape
        new_array = np.zeros(shape=(nrow, ncol))
        new_array[0, :] = gt  # Top row is the same
        new_array[1:, :] = preds  # copy the predictions
    else:  # gt is None
        preds = np.asarray(preds)
        new_array = preds.copy()
        gt_offset = -1
        # Copy the array over since we don't need to concatenate
    gridspec_kw = {}
    if add_second_axis:
        gridspec_kw['width_ratios'] = [2, 1]
    fig, ax = plt.subplots(1, 1 + add_second_axis, figsize=(9, 3), gridspec_kw=gridspec_kw)
    if not add_second_axis:
        ax = [ax]
    divider = make_axes_locatable(ax[0])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.1)
    heatmap = ax[0].imshow(new_array, cmap=plt.cm.get_cmap(cmap_key, 3), interpolation='nearest',
                           vmax=2.5, vmin=-0.5, aspect='auto')  # tab10 is good
    for row_id in range(preds.shape[0] + gt_offset):
        # Add a line between the predictions
        y = row_id + 0.5
        ax[0].axhline(y=y, color='k', linewidth=2)
    lt = [0, 1, 2]  # location ticks
    formatting = plt.FuncFormatter(lambda val, loc: lt[loc])
    fig.colorbar(heatmap, cax=cbar_ax, ticks=[0, 1, 2], format=formatting)
    return fig, ax, heatmap


def join_dict_by_size(min_size, data, rule='forwards', verbose=False):
    """ Join entries in data by key order if less than min_size with rules to handle joining order.

    If the final list in data is not joined, it is placed on the previous remaining dataset.
    Return a dict of joined entries and a dict of join records.
    Note that this is knapsack problem and this solution is non-optimal.

    """

    assert type(data) is dict
    assert type(min_size) is int
    print(data)
    keys = list(data.keys())
    key_move_dict = {key: [] for key in keys}
    if rule == 'forwards':
        keys.sort()
    elif rule == 'reverse':
        keys.sort(reverse=True)
    else:
        raise ValueError(f'key order {rule} was not recognized as an allowed case')
    to_move = []
    move_keys = []
    if verbose:
        print('got keys:', keys)
    removed_keys = []  # for debug
    keys_iter = keys.copy()  # iterator doesn't work well when removing keys while iterating
    for key in keys_iter:  # Already sorted
        if verbose:
            print('key:', key)
        # See if we have any points to move forward
        if len(to_move) > 0:
            if verbose:
                print('Attempting to assign keys from temp list')
            # We have some points to move forward
            data[key].extend(to_move)
            key_move_dict[key].extend(move_keys)
            # Clear the temp lists
            to_move.clear()
            move_keys.clear()
        # attempt joining

        # If the cluster is too small, move it
        if len(data[key]) < min_size:  # we can join
            if verbose:
                print('Attempting pop from key', key)
            to_move.extend(data[key])  # set them aside 
            move_keys.append(key)
            removed_keys.append(key)
            move_keys.extend(key_move_dict[key])  # move the values we have already clustered
            key_move_dict.pop(key)
            data.pop(key)  # Remove key from dict
            keys.remove(key)  # Remove key from sorted list
            if verbose:
                print('keys remaining', keys)

    # We may have data leftover
    if len(to_move) > 0:  # don't loose data
        target = keys[-1]  # put them on the last list we didn't remove 
    for key in keys:
        key_move_dict[key].append(key)
    if verbose:
        print(data)
    return data, key_move_dict
