from typing import Dict, Tuple, List
from numpy.typing import NDArray
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    from .RealData import RealData
except ImportError:
    from RealData import RealData


def distmat_to_csv(matrix: np.ndarray) -> pd.DataFrame:
    assert len(matrix.shape) == 2, 'Must have a matrix'
    assert matrix.shape[0] == matrix.shape[1], 'Matrix must be square'
    assert np.max(np.abs(np.diag(matrix))) == 0, 'Matrix diagonal must be 0'
    assert np.min(matrix) == 0, 'Cannot have non-negative entries'
    np.allclose(matrix - matrix.T, np.zeros_like(matrix)), 'Must be symmetric'
    N = matrix.shape[0]
    data = {'label_1': [], 'label_2': [], 'dist': []}
    for i in range(N):
        dummy_date = '01012000'  # Clustering parser is expecting a date
        key = '|' + dummy_date
        for j in range(i):
            data['label_1'].append(str(i) + key)  # str(i) is the sequence name
            data['label_2'].append(str(j) + key)
            data['dist'].append(matrix[i, j])
    return pd.DataFrame(data)


def parse_wide_data(data: str) -> RealData:
    """
    A utility function for loading data directly from the mega interface to the HIV simulator.

    Args:
        data: str, path to `.mat` data.

    Returns: `RealData.RealData` object.

    """

    test_data = RealData(data, source_type='synth')
    # This file may not be cleaned properly - need to reset labels
    label_new = np.zeros_like(test_data.labels)
    for new, label in enumerate([0, 2, 10]):  # labels we are going to find
        label_loc = label == test_data.labels
        label_new[label_loc] = new
    test_data.labels = label_new

    return test_data


def predict_strict(data: dict, cluster_sizes: List[int]) -> NDArray:
    """ Use a strict prediction rule (include all found) on HIV-Trace JSON data.

    If a sequence is part of a predicted cluster (of any size),
    consider it to be an outbreak.

    We need to know the predictions and the true labels (0,1, or 2)

    """
    preds = [int(ID) for ID in data['Nodes']['id']]  # predict the zeros
    activity_preds = np.zeros(sum(cluster_sizes), dtype=bool)  # apply the r-sorting to labels
    activity_preds[preds] = True  # Set predictions to true
    # Get the true labels
    # outbreak_labels = true_labels == 0  # True = outbreak

    return activity_preds


def trace_clusters(data: dict, cluster_sizes: list[int]) -> Tuple[Dict[int, list], List[Dict[int, int]]]:
    """ Parse the HIV-Trace json to a list of clusters within each "true" cluster.

    Reconstruct the clusters found by HIV-Trace within each "true" cluster from the data. HIV-Trace
    provides a flattened set of cluster ID's for each grouped infection.

    Args:
        data: HIV-Trace JSON in dict format
        cluster_sizes: list of (ordered) cluster sizes

    Returns:
        dict: key-value pairs of clusters within each groups

    """

    infectious_ids, cluster_ids = extract_cluster_data(data)

    return _trace_clusters(cluster_sizes=cluster_sizes,
                           infectious_labels=infectious_ids,
                           cluster_ids=cluster_ids)


def test_trace_clusters():
    cluster_sizes = [5, 5, 5]
    infectious_labels = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    cluster_ids = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]

    expected_clusters = {0: [0], 1: [], 2: [1, 2]}
    expected_counts = [{0: 5}, {}, {1: 3, 2: 2}]

    found_clusters, found_counts = _trace_clusters(cluster_sizes=cluster_sizes,
                                                   infectious_labels=infectious_labels,
                                                   cluster_ids=cluster_ids)
    assert type(found_clusters) is dict
    assert type(expected_counts) is list
    assert found_counts == expected_counts
    assert expected_clusters == found_clusters


def extract_cluster_data(data: dict) -> Tuple[list[int], list[int]]:
    """
    Extract cluster data from HIV-Trace.

    Return both the index of clustered infections and their cluster labels from the HIV-trace output.

    ```python
    infectious_ids, cluster_ids = extract_cluster_data(data)
    ```

    Args:
        data: HIV-Trace output

    Returns:
        view of clustered sequence ids,
        view of list of cluster ids for each sequence.

    """

    infectious_labels = [int(ID) for ID in data['Nodes']['id']]
    try:
        cluster_ids = data['Nodes']['cluster']['values']
    except TypeError:
        cluster_ids = data['Nodes']['cluster']
        cluster_ids = list(map(lambda x: x-1, cluster_ids))
    return infectious_labels, cluster_ids


def _trace_clusters(cluster_sizes: list[int], infectious_labels: list, cluster_ids: list, *args,
                    **kwargs) -> Tuple[Dict[int, list], List[Dict[int, int]]]:
    """
    Reconstruct the clusters for each image.
    Args:
        cluster_sizes:
        infectious_labels:
        cluster_ids:

    Returns:
        dict of discovered clusters represented in each original cluster
        dict of sizes of each discovered cluster

    """

    position = 0
    left, right = 0, 0
    found_counts = []
    for cluster_id, cluster_size in enumerate(cluster_sizes):
        cluster_summation = defaultdict(lambda: 0)  # on new key, set total to 0

        right += cluster_size
        while position < right:  # traverse the container once
            if position in infectious_labels:
                loc = infectious_labels.index(position)
                label = cluster_ids[loc]
                cluster_summation[label] += 1  # initialized to 0 if not found

            position += 1

        found_counts.append(dict(cluster_summation))
    found_clusters = {idx: list(v.keys()) for idx, v in enumerate(found_counts)}
    return found_clusters, found_counts


def predict_largest_k(data: dict, k: int, cluster_sizes: list[int]) -> NDArray:
    """
    Predict the lragest `k` clusters within each known outbreak as active, disregarding smaller clusters.

    Args:
        data: HIVTrace output
        k: Largest number of clusters per outbreak to include.
        cluster_sizes: list[int] how large is each cluster

    Returns:

    """

    # Parse the dict into the correct tree structure
    found_clusters, found_counts = trace_clusters(data=data, cluster_sizes=cluster_sizes)

    trimmed_clusters = []
    infectious_ids, cluster_ids = extract_cluster_data(data)

    for ddict in found_counts:
        if len(ddict) < k:
            trimmed_clusters.append(list(ddict.keys()))  # all the keys we found are outbreak
            continue  # go to next position
        # Use `sorted` to order keys based on item
        sizes = [key for key, value in sorted(ddict.items(), key=lambda item: item[1])]
        trimmed_clusters.append(sizes[:k])  # checked above to ensure that this is valid slice index

    # Now we need to use the clustering to generate labels
    labels = np.zeros(sum(cluster_sizes))  # prediction labels
    left, right = 0, 0
    for cluster_index, cluster_size in enumerate(cluster_sizes):
        right += cluster_size  # so [left, right) contains our clusters
        # Add if an infection is within an accepted cluster
        for infection in range(left, right):
            if infection in infectious_ids:
                inf_loc = infectious_ids.index(infection)
                infection_type = cluster_ids[inf_loc]  # Get the type of the infection
                if infection_type in trimmed_clusters[cluster_index]:  # cluster ID good enough?
                    labels[infection] = True  # set the infection to True
                else:  # outbreak prediction isn't within a permitted cluster
                    pass
            else:  # not predicted outbreak
                pass

    return labels


def convert_true_labels(true_labels: List[NDArray]) -> List[NDArray]:
    """
    Flatten and convert a list of labels (0,1,2) into a 0/1 label.

    Args:
        true_labels List[NDArray]: of true labels

    Returns:
        NDArray - flattened labels
    """

    new_labels = []
    for label in true_labels:
        converted_labels = label == 0
        new_labels.append(converted_labels)
    return new_labels
