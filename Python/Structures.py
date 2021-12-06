import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hclust
import tensorflow as tf
from scipy.io import loadmat
from scipy.spatial.distance import squareform
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class CompilerParameters(object):
    """
    The parameters that will be passed to the Keras model compiler.
    metrics can be either a single string ('acc') or a list of metrics from Keras.
    Additional metrics need to be imported.
    """

    def __init__(self, optimizer=None,
                 loss='categorical_crossentropy',
                 metrics=None):
        if metrics is None:  # avoid default mutable
            metrics = ['acc']
        self.loss = loss
        self.metrics = metrics
        self._lr_schedule = PiecewiseConstantDecay(boundaries=[15], values=[0.00005,
                                                                            0.00001])
        # ADAM uses a dynamic LR, so this adjustment is not terribly important - improves stability
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._lr_schedule, amsgrad=True)
        else:
            self.optimizer = optimizer

    def set_optimizer(self, optimizer):
        """ Override the default optimizer with something new"""
        self.optimizer = optimizer


class ModelGeometry(object):
    """A utility object to store the variables needed to initialize a Keras NN"""

    def __init__(self, _pairmat):
        self.dimensions = _pairmat.pairwise_mats.shape[1:3]  # Get the image size
        self.input = np.append(self.dimensions, [1])  # Get the image size in 3D coordinates
        self.output = _pairmat.train.categorical_labels.shape[1]


class NNModel(object):
    """A wrapper class for tf.keras Models. Contains methods to compile and visualize models."""

    def __init__(self, model_geometry):
        if not isinstance(model_geometry, ModelGeometry):
            raise TypeError('Argument 1 must be a ModelGeometry object')
        input_layer = tf.keras.Input(shape=model_geometry.dimensions)

        outputs = Dense(units=model_geometry.output)(input_layer)
        self.NN = tf.keras.Model(input_layer, outputs)
        self.compiled = False
        # We need to hold on to generators at training time
        self.generator = None
        self.np_generator = None
        self.test_generator = None
        self.history = None

    def compile_model(self, compiler_parameters):
        """ Provide a compiler_parameters object to compile the object. """
        if not isinstance(compiler_parameters, CompilerParameters):
            raise TypeError('Argument 2 must be a CompilerParameters object')
        self.NN.compile(optimizer=compiler_parameters.optimizer,
                        loss=compiler_parameters.loss,
                        metrics=compiler_parameters.metrics)
        self.compiled = True

    def plot(self, filename, show_shape=True):
        plot_model(self.NN, to_file='../Model/' + filename + '.png', show_shapes=show_shape)

    def train(self, pair_mat, training_parameters, test_pair_mat=None,
              logdirpath='../Model/TensorBoard/'):
        """Train the model with the specified data and training parameters. Test data optional.

        Train the model using provided information using the default keras training routine.
        Tensorboard logging is enabled by default. Verbosity is set at 2 and is suitable to
        be captured.

        """

        if not isinstance(pair_mat, PairMat):
            raise TypeError('Argument 2 must be a PairMat data object')
        if not self.compiled:
            raise ValueError('The model must be compiled before it can be trained')
        if not isinstance(training_parameters, TrainingParameters):
            raise TypeError('Argument 4 must be a TrainingParameters object')
        if test_pair_mat is not None and not isinstance(test_pair_mat, PairMat):
            raise ValueError('Argument 3 must be either None or a PairMat Data object.')
        assert type(logdirpath) is str

        logdir = logdirpath + datetime.now().strftime("MD%m%d_HMS%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, embeddings_freq=5,
                                                              write_images=True)

        self.generator = tf.keras.preprocessing.image.ImageDataGenerator()
        self.np_generator = tf.keras.preprocessing.image.NumpyArrayIterator(x=pair_mat.pairwise_mats,
                                                                            y=pair_mat.train.categorical_labels,
                                                                            image_data_generator=self.generator,
                                                                            data_format='channels_last',
                                                                            batch_size=training_parameters.batch_size,
                                                                            shuffle=True)
        if test_pair_mat is None:
            self.history = self.NN.fit(self.generator.flow(x=pair_mat.pairwise_mats,
                                                           y=pair_mat.train.categorical_labels,
                                                           batch_size=training_parameters.batch_size),
                                       steps_per_epoch=pair_mat.pairwise_mats.shape[0]//training_parameters.batch_size,
                                       verbose=2)
        else:
            self.history = self.NN.fit(
                x=self.np_generator,
                steps_per_epoch=pair_mat.pairwise_mats.shape[0] // training_parameters.batch_size,
                epochs=training_parameters.epoch,
                validation_data=(test_pair_mat.pairwise_mats, test_pair_mat.train.categorical_labels),
                callbacks=[tensorboard_callback],
                validation_steps=test_pair_mat.pairwise_mats.shape[0] // training_parameters.batch_size, verbose=2)

    def save_model(self, model_name, prefix='../Model/'):
        """ Execute the save routine. A convenience method that interfaces to the core save method."""
        label = prefix + model_name + '.h5'
        self.NN.save(label)

    def import_model(self, model_name):
        """We want to import a pre-trained model and use it"""
        label = '../Model/' + model_name + '.h5'
        self.NN = tf.keras.models.load_model(label)
        self.compiled = True

    def predict(self, x):
        return self.NN.predict(x=x)


class TrainingParameters(object):
    """Store the hyper-parameters for training for a model"""

    def __init__(self):
        self.epoch = 10
        self.batch_size = 64
        self.randomize = True
        self.validation_steps = 300

    def __str__(self):
        line_1 = f'epochs: {self.epoch}\n'
        line_2 = f'batch_size: {self.batch_size}\n'
        line_3 = f'validation steps: {self.validation_steps}\n'
        str_state = 'enabled' if self.randomize else 'disabled'
        line_4 = f'Randomize flag {str_state}\n'
        return line_1 + line_2 + line_3 + line_4


class MultipleModel(object):
    """Collect multiple trained models into a OVA model"""

    def __init__(self):
        self.models = None
        self.length = 0
        self.filled = False
        self.filepath = None
        self.file_ = None

    def import_by_name(self, models):
        """ Import a collection of models """
        if not all(isinstance(models[x], NNModel) for x in range(len(models))):
            raise TypeError('At least one of the models input is not of the correct type')
        self.models = self.models.extend(models)
        self.length = + self.length + len(models)
        self.filled = True
        self.file_ = []

    def import_from_directory(self, target_dir):
        """
        Import all models in a target directory. Model will be added in the
        order in which they are enumerated via os.listdir(), the
        behavior of which defaults to alphabetical ordering. 
        """
        if not isinstance(target_dir, str):
            raise TypeError('The target directory must be entered as a string')
        self.filepath = target_dir
        # print( self.filepath)
        # print('Flag 1!')
        # print('sorted files are:', sorted(os.listdir(self.filepath)))
        # files = None
        # TODO Check that the files we find are hdf5 format
        if self.filepath[-1] == '/':
            # Remove the trailing slash
            self.filepath = self.filepath[:-1]

        self.file_ = [os.path.join(self.filepath, file) for file in sorted(os.listdir(self.filepath)) if
                      os.path.isfile(os.path.join(self.filepath, file))]
        print(self.file_)
        if self.file_ is None:
            message = 'The file path:' + self.filepath + ' does  not contain any files'
            raise ValueError(message)
        self.models = [None] * len(self.file_)  #
        for index in range(len(self.file_)):
            file = self.file_[index]
            print(file)
            self.models[index] = load_model(file)
        if self.models is not None:
            self.filled = True
            self.length = len(self.models)
        return self.filled

    def predict(self, x, return_vector=True, steps=1, return_probabilities=False):
        """
        Predicts labels using all models with a winner-take-all voting scheme.
        :param x:
        :param return_vector:
        :param steps: [optional] int, ignored unless x is a generator object
        :return:

        """

        if type(x) is np.ndarray:
            pred_array = np.empty((x.shape[0], len(self.models)))
            for model_id in range(len(self.models)):
                pred_array[:, model_id] = np.argmax(self.models[model_id].predict(x=x), axis=1)
        else:  # assume a generator expression has been input
            pred_list = []
            for model_id in range(len(self.models)):
                pred_list.append(np.argmax(self.models[model_id].predict(x=x, steps=steps), axis=1))
                pred_list[model_id].reshape(1, pred_list[model_id].shape[0])
            pred_array = np.empty((pred_list[0].shape[0], len(self.models)))

            # Cleanup now that we know how much data we got
            for model_id in range(len(self.models)):
                pred_array[:, model_id] = pred_list[model_id]

        pred_returns = np.empty(pred_array.shape[0])
        """ Make a prediction with the multiple models"""
        for index in range(pred_returns.shape[0]):
            pred_returns[index] = np.argmax(np.bincount(pred_array[index, :].astype('int8')))
        if return_probabilities:
            return pred_array, pred_returns

        if return_vector:
            return pred_returns
        else:
            return pred_array
        # We want to evaluate each model at x and then assign 'points' based on each prediction,
        # Then return the prediction with the most points

    def evaluate(self, x_true, y_true, return_total=False):
        """ Evaluate the model accuracy on data"""
        # return_total: return a tuple (number of correct, total attempted)
        assert self.filled, 'We must have a model to evaluate'
        y_pred = self.predict(x=x_true, return_vector=True)
        print('y_pred shape')
        print(y_pred.shape)
        print('y_pred is')
        print(y_pred)
        values = np.equal(y_pred, np.argmax(y_true, axis=1))
        print('values is')
        print(values)
        if return_total:
            return (np.sum(values), len(values))
        else:
            return np.sum(values) / len(values)

    def evaluate_on_each_model(self, x_true, y_true):
        """ Evaluate the data on each model. """
        assert self.filled, 'We must have a model to evaluate'
        y_pred = self.predict(x=x_true, return_vector=False)  # A matrix of predictions for each model
        # print('y_pred.shape is')
        print(y_pred.shape)
        y_model_acc = np.zeros(len(self.models))
        y_true = np.argmax(y_true, axis=1)
        for modelNo in range(0, self.length):
            y_model_acc[modelNo] = np.sum(np.equal(y_true, y_pred[:, modelNo])) / y_true.shape[0]

        return y_model_acc


class PairMat(object):
    """ A class to store pairwise distance matrices including annotations and indices.
    """

    def __init__(self, file_name, num_classes=None, method='OLO'):
        """
        param: file_name: path-like object to data file
        param: num_classes: (optional) override and increase label creation size. 
        param: method (optional) string, which cluster method to use. Default is OLO
              availible options are 'None', 'HC' and 'OLO'.
        param: HC_method: string: valid method for 
        """

        self.vectors = None
        self.filepath = str(file_name)
        if os.path.isfile(self.filepath):
            self.valid = True
        else:
            message = 'The file path:' + self.filepath + ' is not valid'
            raise ValueError(message)

        # We have confirmed that the file is valid, now fill in the fields
        self.data = loadmat(self.filepath)
        self.pairwise_mats = np.zeros(shape=(self.data['matrices'].shape + (1,)))
        self.pairwise_mats[:, :, :, 0] = self.data[
            'matrices']  # Input layer is expecting a samples x rows x col x channels
        self.labels = np.rint(self.data['indexes'][:, :2])  # skip lambda
        self.train = TrainingLabels(self, num_classes=num_classes)
        self.log_text = str(self.data['log'])
        self.NPop = self.data['NPop']
        self.R0_vals = self.data['values'][:, 0]
        delattr(self, 'data')  # Remove the initial data object loaded in\\
        if method == 'OLO':
            print('Beginning to leaf-order optimize images')
            self.leaf_order_optimize()
            print('Finished leaf-order optimizing images')
        elif method == 'HC':
            print('Beginning to perform hierarchical clustering')
            self.hc()
            print('Finished hierarchical clustering')
        else:
            print(f'Skipping order optimization step with option {method}')

    def to_vectors(self):
        if self.vectors is None or self.vectors.ndim != 2:
            self.vectors = self.pairwise_mats.copy()
            self.vectors = self.vectors.reshape((self.pairwise_mats.shape[0], -1))
        else:
            pass  # don't need to reshape

    def leaf_order_optimize(self, method=hclust.ward):
        """
        Reorder the leaves of the images in place. Default single-linkage
        ordering method computes a minimum spanning tree via nearest point algorithm.
        """
        for row in range(self.pairwise_mats.shape[0]):
            if row % 1000 == 0:
                print('Reached row', row)
            #    print(self.pairwise_mats[row,:,:,0])
            distvec = squareform(self.pairwise_mats[row, :, :, 0])
            linkage_map = method(distvec)
            optimal_linkage_map = hclust.optimal_leaf_ordering(Z=linkage_map, y=distvec)
            order = hclust.leaves_list(optimal_linkage_map)
            self.pairwise_mats[row, :, :, 0] = self.pairwise_mats[row, order, :, 0]
            self.pairwise_mats[row, :, :, 0] = self.pairwise_mats[row, :, order, 0]

    def hc(self, method='ward'):
        for row in range(self.pairwise_mats.shape[0]):
            if row % 1000 == 0:
                print('Reached row', row)
            distvec = squareform(self.pairwise_mats[row, :, :, 0])
            linkage_map = hclust.linkage(distvec, method=method)
            order = hclust.leaves_list(linkage_map)
            self.pairwise_mats[row, :, :, 0] = self.pairwise_mats[row, order, :, 0]
            self.pairwise_mats[row, :, :, 0] = self.pairwise_mats[row, :, order, 0]

    def show(self, image_id, cmap='viridis', ax=None, fig=None):
        """ Visualize a matrix from the stack of data by index."""
        if ax is None and fig is None:
            fig, ax = plt.subplots(1)
        image = np.squeeze(self.pairwise_mats[image_id, :, :, 0])
        vmax = np.max(image)
        image = np.squeeze(image)
        main_IM = ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(main_IM, ax=ax)
        return fig, ax


class TrainingLabels(object):
    """
    A helper class for storing the labels used for CNN training.
    Vectorized training labels are stored in `categorical labels`.
    """

    def __init__(self, pair_mat, num_classes=None):
        self.long_labels = np.array(pair_mat.data['labels'], dtype=np.int8)
        self.indexes = np.array(pair_mat.data['indexes'][:, :2], dtype=np.int8)  # Skip Lambda parameter in position 1
        self.labels = np.rint(pair_mat.data['labels'])
        if num_classes is None:
            self.categorical_labels = to_categorical(self.labels - 1)
        else:
            self.categorical_labels = to_categorical(self.labels - 1, num_classes=num_classes)

        self.values = np.array(pair_mat.data['values'])
        self.labels_max = np.amax(np.rint(pair_mat.data['indexes'][:, :2]), axis=0)
        self.labels_shape = np.array([len(np.unique(np.rint(pair_mat.data['indexes']), axis=0)), 1])

        def make_labels():
            iter_counter = 0
            offset_labels = self.indexes.copy()
            for col in range(1, self.indexes.shape[1]):
                iter_counter = iter_counter + np.amax(self.indexes[:, col - 1])
                offset_labels[:, col] = np.add(
                    self.indexes[:, col], np.full((1, self.indexes.shape[0]), iter_counter))
            return offset_labels

        self.labels = make_labels()
