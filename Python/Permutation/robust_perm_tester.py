"""
rubst_perm_test.py
Michael Kupperman

Validate a model/image pair against pairwise permutations.

The internals are built using factoradic representations. See:
https://en.wikipedia.org/wiki/Factorial_number_system
for details.
"""

import itertools
import math

import numpy as np


def index_to_factoradic(value, num_elem=None):
    """ Convert the int:index to a factoradic representation in a list, indexed 0 to n-1
    Args"
        value: the integer to convert to mixed radix format
        num_elem: Ignored. For API compatability with `factoradic_to_permutation`.
    Returns:

    """
    remainder = value % 1
    value = value - remainder
    factorial_rep = [remainder]
    index = 2
    while value > 0:
        remainder = value % index
        value = value // index
        index += 1
        factorial_rep.append(remainder)

    return factorial_rep


def factoradic_to_index(factoradic):
    """ Convert a factoradic to the index."""
    value = 0
    for idx in range(len(factoradic)):
        value += factoradic[idx] * math.factorial(idx)
    return value


def factoradic_to_permutation(factoradic, num_elem):
    """ Convert a factoradic to a permutation of `num_elem` elements.
        The factoradic is returned in the reverse order (read right-to-left)
    """
    elements = [val for val in range(num_elem)]
    permutation = np.empty(shape=(num_elem,), dtype=np.int)
    idx = 0
    while len(factoradic) > 0:
        val = factoradic.pop()
        permutation[idx] = elements[val]
        elements.remove(elements[val])
        idx += 1
    for pos in range(idx, num_elem):
        permutation[pos] = idx
        idx += 1
    return permutation


def hone_factoradic_size(k):
    """ Find the smallest n such that k < n!
        Mostly implemented for debug purposes. """
    assert type(k) is int, 'k must be an integer'
    assert k > 0, 'k must be positive'
    n = 1
    while True:
        if math.factorial(n) > k:
            return n
        else:
            n += 1


def permutation_to_factoradic(permutation):
    """ Convert a permutation to a factoradic representation.

    A utility function for checking factoradic representations.

    Args:
         permutation (np.ndarray):
    Return:
       list[int]
    """
    x = []
    na = [val for val in range(permutation.shape[0])]
    for perm_val in np.nditer(permutation):
        permutation_index = perm_val.item()
        factoradic_value = na.index(permutation_index)  # get position
        x.append(factoradic_value)
        na.remove(permutation_index)

    x.reverse()  # flip to reverse order
    return x


def reorder(image, local_perm, global_perm, image_size):
    """ Reorder an image using two permutations

    Use global_perm to determine which indicies will be reordered.
    Use local_perm to determine the reordering to apply.

    Args:
        image: A square numpy array to reorder,
        local_perm: Which permutation to use to reorder
        global_perm: Which indices to reorder
        image_size:

    Returns:

    """
    # Modify global permutation with local permutation
    global_perm_new = global_perm[local_perm]
    id_map = np.arange(start=0, stop=image_size, dtype=np.int8)
    id_map[global_perm] = global_perm_new
    # Copy the image to avoid side effects
    new_image = image[:, id_map, :, :]
    new_image = new_image[:, :, id_map, :]

    return new_image


def generator_image_permutator(image, batch_size, num_swap, batches_sent=0, abs_idx=None):
    """ A generator expression for permutations. Generates batches of permuted images.
    Parameters abs_idx and batches_sent are for debug purposes only. The last batch may be smaller
    than `batch_size`.
    """

    image_tmp = image.copy()
    image_size = image.shape[1]
    if type(batch_size) == np.ndarray:
        batch_size = batch_size.item(0)
    if type(num_swap) == np.ndarray:
        num_swap = num_swap.item(0)
    im_total = (math.factorial(num_swap) - 1) * math.factorial(image.shape[1]) / \
               (math.factorial(num_swap) * (math.factorial(image.shape[1] - num_swap)))
    while True:
        im_seen = 0
        if batches_sent != 0:
            batches_sent = 0
        # Generate iterator of global permutations
        global_permutation_generator = itertools.combinations(range(image_size), num_swap)
        abs_idx = 0  # for constructing batches
        batch = np.empty(shape=(batch_size, image_size, image_size, 1))
        # debug flag flag to make sure that the last batch is evaluated even if it is too small
        last_yield = False
        gperm_idx = 0
        for gperm in global_permutation_generator:
            gperm_idx += 1
            for index in range(1, math.factorial(num_swap)):
                last_yield = True
                lperm = factoradic_to_permutation(index_to_factoradic(value=index, num_elem=num_swap),
                                                  num_elem=num_swap)
                image_tmp2 = reorder(image_tmp, local_perm=lperm, global_perm=np.array(gperm), image_size=image_size)

                batch[abs_idx, :, :, :] = image_tmp2
                im_seen += 1
                abs_idx += 1
                if abs_idx == batch_size:
                    last_yield = False
                    batches_sent += 1
                    yield batch
                    abs_idx = 0
        # If you made it this far , you're out of data
        if abs_idx != 0:
            # Drop extra entries we haven't overwritten
            batch = batch[0:abs_idx, :, :, :]
            batches_sent += 1
            yield batch
            batches_sent = 0  # reset the state


def permutation_test(data, model, real_labels, batch_size=32, num_swap=2, num_im=10):
    """
    Perform permutation robustness of num_swap elements of model using .predict method
    with the first num_im slices from data. If there are less than num_im images, all 
    images are processed. 
    
    Returns:
        acc_vals: Accuracy for each image
        right_imid: Images correctly initially predicted
        incorrect_imid: Images incorrectly initially predicted
    """
    right_imid = []
    incorrect_imid = []
    acc_vals = []
    # for any allowed, .shape[1] will always be non-one
    if type(real_labels) is int:
        real_labels = [real_labels]

    for im_id in range(data.shape[0]):
        if im_id > num_im:
            break
        if len(data.shape) == 4:
            # in correct shape
            im = data[im_id, :, :, :]
            im = im[np.newaxis, :, :, :]
        elif len(data.shape) == 3:
            # 3d tensor, add 4th dim
            im = data[im_id, :, :, np.newaxis]
            im = im[np.newaxis, :, :, :]
        elif len(data.shape) == 2:
            im = data[np.newaxis, :, :, np.newaxis]
        else:
            raise ValueError('Input tensor must be 2, 3, or 4 dimensions')
        mp = model.predict(im)
        if mp == real_labels[im_id]:
            right_imid.append(im_id)
        else:
            incorrect_imid.append(im_id)
        variants = num_swap  # k
        num_images = (math.factorial(variants) - 1) * (math.factorial(im.shape[1]) /
                                                       (variants * (math.factorial(im.shape[1] - variants))) - 1)
        imgen = generator_image_permutator(image=im, batch_size=batch_size, num_swap=num_swap)
        steps = np.ceil(num_images / batch_size).astype(np.int64)
        preds = model.predict(imgen, steps=steps)
        rlabel = real_labels[im_id]
        acc = np.average(np.equal(rlabel, preds))
        if acc == 1.0:
            # Check the result via errors, possible loss of significance via floating-point errors
            eq = np.equal(rlabel, preds)
            acc = 1 - ((preds.shape[0] - np.sum(eq)) / preds.shape[0])
        acc_vals.append(acc)
    return acc_vals, right_imid, incorrect_imid
