import os
from os.path import join, isfile
import RealData as RD
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

activity_colors = ListedColormap(['lightcoral', 'sandybrown', 'white', 'firebrick', 'black'])
colors_ = ['lightcoral', 'sandybrown', 'firebrick', 'black']

class TimeSeries(object):
    def __init__(self, scale_factor=1):
        """
        class to interface with user and handle many ReadData objects.
        Requires `self.initialized == True` to access most methods.
        """
        self.initialized = False
        self.files = None
        self.data = None
        self.num_slices = None
        self.file_idx = None
        self.last_labels = None
        self.last_preds = None
        self.scale_factor = scale_factor
        self.labels = None
        
    def load_series_from_dir(self, directory, splitchar='-'):
        """
        Load in all time series from directory.
        Optional splitchar determines which chars to split to get idx from file name.
        First 
        Under default behavior:
        good filenames:
        `DataFile-1.mat`
        `MajorEpiSample-01-EU.eps`
        
        Bad filenames:
        `DataFile-2012-12.eps`  Read in as idx 2012
        `dataFile-12.big.mat` no idx recovered,
        """
        self.files =  [file for file in sorted(os.listdir(directory)) if isfile(join(directory, file))]
        self.file_idx = []
        # We may not have gotten files in the right order
        for file in self.files:
            # First get the integer index
            v = file.strip('.mat').split(splitchar)
            flag = False
            for char in v:
                try:
                    if not flag:
                        # we only do this once
                        self.file_idx.append(int(char))
                        flag = True
                except ValueError:
                    # Leap before you look design
                    pass
            if not flag:
                raise ValueError('An idx from file ' + file + ' was not obtained.' )
        # Now sort the lists using the key ordering we just got
        self.files  =[ file for _, file in sorted(zip(self.file_idx, self.files))]
        self.file_idx = sorted(self.file_idx)
        #self.files = sorted(self.files, key=lambda x: self.file_idx[x])
        #self.file_idx = sorted(self.file_idx)
        self.data = [RD.RealData(join(directory, file), source_type='real', scale_factor=self.scale_factor) for file in self.files]
        self.labels = []
        for data in self.data:
            for item in data.row_labels:
                if type(item.item()) is str:
                    self.labels.append(item.item())
                else:
                    self.labels.append(item.item().item())
        self.labels = sorted(list(set(self.labels)))
        self.ordered_labels = self.data[-1].row_labels.copy()
        self.initialized = True
        
    def evaluate(self, model, window_size, choice_method, cluster_method='None', second_clustering_method='None'):
        """
        Evaluate the time series sequence of images. 
        """
        preds_list = []
        label_list = []
        map_list = []
        
        # Loop over the images, throw warning for images too small
        for idx, data in zip(self.file_idx, self.data):
            if data.matrices.shape[1] < window_size:
                warn(f'On data {idx}, the number of samples wth shape {data.matrices.shape} is less than the window size')
                continue
            preds, sort_map, labels = data.predict(model=model, window_size = window_size, choice_method=choice_method,
                                         cluster_method=cluster_method, second_clustering_method=second_clustering_method,
                                        return_labels=True, return_map=True)
            preds_list.append(preds)
            label_list.append(labels)
            map_list.append(sort_map)
        # For each image: store the index, prediction (0,1,2 encoded), labeldict
        
        # Return list of lists
        self.last_preds = preds_list
        self.last_labels = label_list
        return preds_list, label_list, map_list
        
def trace_history(history, idx):
    """ Determine all labels applied to an individual idx through time"""
    labels_applied = []  # what label was applied? 0,1, or 2?
    times = []  # store the indexes where we found the sample
    for labeldict, num_slice in zip(history, range(len(history))):
        # Loop over list of histories (dicts)
        for index in labeldict.keys():
            if idx in labeldict[index]:
                labels_applied.append(index)
                times.append(num_slice)
    return labels_applied, times


def _plot_preds(preds, start=0, end=None, step=10, fig=None, ax=None, cbar=True):
    """
    Generate an unlabeled plot using preds. A row is an individual, a column is a sample/time point.
    """
    image = build_matrix_from_preds(preds)
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    hot = plt.cm.get_cmap('hot')
    print(image.shape)
    if end == None:
        predmap = ax.imshow(image, cmap=hot.reversed())
    if end is not None:
        xend = np.min([end, image.shape[0]])
        yend = np.min([end, image.shape[1]])
        print([start, xend, start, yend])
        predmap = ax.imshow(image[start:xend, start:yend], cmap=hot.reversed())
        ax.set_xticks(np.arange(0, yend-start, step=step))
        ax.set_xticklabels(labels=np.arange(start, yend, step=step))
        ax.set_yticks(np.arange(0, xend-start, step=step))
        ax.set_yticklabels(np.arange(start, xend, step=step))
    if cbar:
        fig.colorbar(predmap, ax=ax)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Patient')
    return ax, fig

def plot_activity(preds, sort_map_list, label_list, window_size=2, active_length_cutoff=3, disable_inactive=True,
                 start=None, end=None, step=10, ax=None,fig=None, cbar=False, legend=True):
    image = build_activity_matrix_overlay(preds=preds, sort_map_list=sort_map_list, label_list=label_list,
                                          window_size=window_size, active_length_cutoff=active_length_cutoff,
                                          disable_inactive=disable_inactive)
    
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    print(image.shape)
    cblabels = ['Reactivated outbreak', 'Inactive outbreak', 'NA', 'Active outbreak', 'Endemic']
    cblabels_ = ['Reactivated epidemic', 'Inactive epidemic', 'Active epidemic', 'Endemic']
    cblabels__ = ['Reactivated outbreak', 'Inactive outbreak', 'Active outbreak', 'Endemic']
    if end == None:
        # set interpolation to nearest - weird white patches can sometimes pop up...
        predmap = ax.imshow(image, cmap=activity_colors, vmin = -3.5, vmax = 1.5, interpolation='none')
    if end is not None:
        xend = np.min([end, image.shape[0]])
        yend = np.min([end, image.shape[1]])
        print([start, xend, start, yend])
        predmap = ax.imshow(image[start:xend, start:yend], cmap=activity_colors, interpolation='none')
        ax.set_xticks(np.arange(0, yend-start, step=step))
        ax.set_xticklabels(labels=np.arange(start, yend, step=step))
        ax.set_yticks(np.arange(0, xend-start, step=step))
        ax.set_yticklabels(np.arange(start, xend, step=step))
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar_obj = fig.colorbar(predmap, cax=cax, ticks = [-3, -2, -1, 0, 1])
        cbar_obj.ax.set_yticklabels(cblabels)
    if legend:
        ph = [ax.plot([], label=cblabels_[i], marker='s', linestyle=(0,(1,100)), mfc=colors_[i], mec=colors_[i],
                      c='w', markersize=50)[0] for i in range(4)]
        ph.reverse()
        cblabels_.reverse()
        ax.legend(ph, cblabels_, loc='lower left')
    ax.set_xlabel('Outbreak progression')
    ax.set_ylabel('Collected sequence')
    return ax, fig

    
def build_matrix_from_preds(preds):
    cols = len(preds)
    rows = preds[-1].shape[0]  # number of 
    image = - np.ones((rows, cols))
    for col in range(cols):  # Loop over each time point of data
        length = preds[col].shape
        length = length[0]
        image[0:length, col] = preds[col]
    return image

def build_activity_matrix_overlay_(preds, sort_map_list, label_list, active_length_cutoff=3, 
                                   disable_inactive=True, window_size=2):
    pred_image = build_matrix_from_preds(preds)
    pred_edited = deepcopy(preds)
    # 
    for pred in pred_edited:
        pred = np.minimum(pred, 1)
    #image = np.minimum(pred_image, 1)  # turn any 2's into 1's
    year_offset = pred_image.shape[0] - pred_image.shape[1] + 1
    sample_years = strip_string_to_year(label_list)
    for person in range(pred_image.shape[0]):
        # loop over each pred and determine if they need to be turned off
        birth_year = sample_years[person]
        for observation_index in range(person, pred_image.shape[1]):
            # get the year to compare against
            cur_year = sample_years[observation_index + year_offset]
            age = cur_year - birth_year
            if age >= active_length_cutoff and preds[observation_index][person]==0:
                #print(image.shape, person, observation_index)
                #image[person, observation_index] = -2
                pred_edited[observation_index][person] = -2


    for person in range(pred_image.shape[0]):
        # loop over each pred and determine if they need to be turned off
        birth_year = sample_years[person]
        for observation_index in range(person, pred_image.shape[1]):
            if pred_edited[observation_index][person] == -2:
                neighbors = get_neighbors(person, sort_map_list[observation_index], window_size)
                print(person, observation_index, neighbors)
                if np.any( np.equal(pred_edited[observation_index][neighbors], 0)):
                    pred_edited[observation_index][person] = -3
                    #image[person, observation_index] = -3
    image = build_matrix_from_preds(pred_edited)
    return image

def build_activity_matrix_overlay(preds, sort_map_list, label_list, active_length_cutoff=3, 
                                  disable_inactive=True, window_size=2):
    spreds = deepcopy(preds)
    year_offset = spreds[-1].shape[0] - len(preds) - 1
    sample_years = strip_string_to_year(label_list)
    #print(len(sample_years), sample_years)
    for idx in range(len(spreds)):
        # clip any 2's into 1's
        spreds[idx] = np.minimum(spreds[idx], 1)
    
    #find active individuals to turn off
    for sample_index, (sample, sort) in enumerate(zip(spreds, sort_map_list)):
        #print(sample_index, sample_index + year_offset)
        sample_age = sample_years[sample_index] # + year_offset]
        #print(sample_age, sample.shape[0]-1)f
        for person in range(sample.shape[0]-1):
            # print(sample_index, person)
            start_age = sample_years[person]
            #print(start_age - sample_age, sample_index, person)
            if sample_age - start_age > active_length_cutoff and sample[person]==0:
                # print(sample_index, person)
                sample[person] = -2

    # See if each individual is near an active outbreak
    for sample_index, (sample, sort) in enumerate(zip(spreds, sort_map_list)):
        for person in range(sample.shape[0]):
            if sample[person] == -2:  # Only proceed if sample label could be switched
                neighbors = get_neighbors(person, sort, window_size)
                if np.any(np.equal(sample[neighbors], 0)):
                    sample[person] = -3
    return build_matrix_from_preds(spreds)

def get_neighbors(index, sort_list, k):
    """
    Given a list of sorted individuals, find `k` closest neighbors on both sides
    
    example: 
    > get_neighbors(3, [2,5,0,3,4,1], 1)
    [5,0,3]
    > get_neighbors(5, [2,5,0,3,4,1], 2)
    [2,5,0,3]
    """
    loc = (sort_list == index).nonzero()[0]  # list is unique
    # print(index, loc, np.maximum(0, loc - k), np.minimum(loc + k, sort_list.shape[0]))
    return sort_list[np.maximum(0, loc - k).item():np.minimum(loc + k, sort_list.shape[0]).item()]

def strip_string_to_year(strings, default_birth_year=1992):
    # Forwards pass 
    years = []
    for label_string in strings:
        tokens = label_string.split('.')
        # print(tokens)
        found_year = False
        year = default_birth_year # default value if we don't know the birth year
        
        for token in tokens:
            try:
                year = int(token)
                if year < 2030 and year > 1980 and not found_year:
                    found_year = True
                    years.append(year)
                    #print(year, token)
                    break
            except ValueError:
                pass
        if not found_year:
            years.append(year)
            
    # Backwards pass
    first_year = np.argmax(np.asarray(years) > 0)
    years[0] = years[first_year]
    for index in range(1, len(years)):
        if years[index] < years[index-1]:
            years[index] = years[index-1]
    return years





