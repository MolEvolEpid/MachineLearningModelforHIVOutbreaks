"""
Test stability of a model on data against permutations of the data using random permutations.


"""

import numpy as np
import math
from tqdm import trange


from collections import namedtuple

def reorder_image(image_size:int, image:np.ndarray,
                  lorder:np.ndarray, gorder:np.ndarray) -> np.ndarray:
    """
    Apply a reordering using the provided permutation. generates a new image.


    Args:
        image_size (int): Passed in for speed.
        image (np.ndarray): Symmetric matrix to apply permutation to

    Returns:
        np.ndarray: _description_
    """
    # Copy the image to have a pure function
    new_image = image.copy()
    # Apply the permutation
    order = np.arange(image_size, dtype=np.int8)
    order[gorder] = gorder[lorder]
    # print(order, image.shape)
    new_image = image[:, order, :, :]
    new_image = new_image[:, :, order, :]

    return new_image


Permutation = namedtuple('Permutation', ['local_', 'global_'])

def get_test_batch(image, batch_size, num_swap, depth):
    # Sample permuted images with replacement.
    image_tmp = image.copy()
    image_size = image.shape[1]
    if type(batch_size) == np.ndarray:
        batch_size = batch_size.item(0)
    if type(num_swap) == np.ndarray:
        num_swap = num_swap.item(0)
    generator = np.random.default_rng()
    # max_number = (math.factorial(image_size)) / (math.factorial(num_swap) * math.factorial(image_size - num_swap))
    # max_number = max_number * (math.factorial(image_size) - 1)  # number of non-identity ways to re-arrange
    batch = np.zeros(shape=(depth, image_size, image_size, 1), dtype=float)
    # Check that we won't get repeats
    assert num_swap <= image_size, f'Cannot swap more elements than there are! Got: {num_swap=} with strict upper bound {image_size=}'
    # Number of possible groups to re-arrange

    # Find `depth` random permutations
    permutations = []
    while len(permutations) < depth:
        sample_global = generator.choice(image_size, size=num_swap, replace=False, shuffle=False)
        sample_local = generator.permutation(num_swap)
        x = Permutation(local_=sample_local.tolist(), global_=sample_global.tolist())
        # print(x, x in permutations, len(permutations))
        # if not (x in permutations):
        # Sample with replacement
        permutations.append(x)

    for index, perm in enumerate(permutations):
        batch[index,:,:] = reorder_image(image=image_tmp, image_size=image_size,
                                         lorder=np.asarray(perm.local_, dtype=np.int8),
                                         gorder=np.asarray(perm.global_, dtype=np.int8))

    return batch

    # terminate_flag = False
    # while not terminate_flag:
    #     # Stop at batch, how many more are possible, or how many do you need to satisfy the depth
    #     batch_depth = min(batch_size, max_number - seen, seen - depth)
    #     print(batch_depth)
    #     terminate_flag = max_number - seen == batch_depth or seen == depth

    #     for index in range(batch_depth):
    #         sample = None
    #         # Loop over sampling until we find a new permutation
    #         while sample in tested:
    #             # Get a random collection of elements.
    #             sample_global = generator.choice(image_size, size=num_swap, replace=False, shuffle=False)
    #             sample_local = generator.permutation(num_swap)
    #             seen += 1
    #             if sample
    #                 batch[index,:,:] = reorder_image(image=image_tmp, lorder=sample_local, gorder=sample_global)
    #                 tested.update(batch_depth)

    #     if terminate_flag:
    #         # Only send out the portion of the batch we filled
    #         # Ok to truncate data structure here - if we
    #         batch = batch[:batch_depth, :,:]

    #     yield batch
    # if terminate_flag:
    #     # We're done. Stop now.
    #     raise StopIteration




def sample_permutation_test(data, model, real_labels, batch_size=32, num_swap=2, num_im=10, depth=1000):

    right_imid = []
    incorrect_imid = []
    acc_vals = []
    tested = 0  # How many permuted images did we attempt
    correct_preds = 0
    consistent_values = []
    for im_id in trange(min(num_im, data.shape[0])):
        # You can't check more values than are in the data
        # First clean the data
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
        # Get the generator
        data_iterator = get_test_batch(image=im, batch_size=batch_size, depth=depth, num_swap=num_swap)
        steps = math.ceil(depth / batch_size)
        preds = model.predict(data_iterator, steps=steps)
        correct = preds == real_labels[im_id]
        consistent = preds == mp
        correct_preds += sum(correct)
        tested += len(correct)
        acc_vals.append(sum(correct)/len(correct))
        consistent_values.append(consistent.mean())

    return acc_vals, consistent_values
