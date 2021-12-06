import os
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.io import loadmat, savemat
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from datetime import datetime
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hclust
import matplotlib.pyplot as plt


class SimpleData(object):
    def __init__(self):
        """
        A lightweight class for storing a collection of images and
        their labels, designed to be easily serializable to and from
        a file. 
        """
        self.labels = None
        self.images = None
    
    def import_from_file(self, file_name):
        """
        Import from a .mat file.
        """
        data = loadmat(file_name)
        self.labels = data['labels']
        self.images = data['images']

    def save(self, file_name):
        data = {}
        data['labels'] = self.labels
        data['images'] = self.images
        savemat(file_name=file_name, mdict=data)        

    def from_PairMat(self, pairmat):
        # assert type(pairmat) is PairMat, f'pairmat should be a PairMat object, got type {type(pairmat)}'
        self.images = pairmat.pairwise_mats
        self.labels = np.argmax(pairmat.train.categorical_labels, axis=1)

    def remove(self, model, mode='correct', score=1):
        """
        Remove images classified by `model.predict` as `mode`.
        Specify `mode='correct'` to remove correctly classified images
        Specify `mode='error'` to remove incorrectly classified images.

        Score is currently not implemented
        """
        if type(model) is MultipleModel:
            for nn in model.models:
                self._remove(model=nn, mode=mode, score=score)
        else:
            self.remove(model=model, mode=mode, score=score)

    def _remove(self, model, mode, score=1):
        to_keep = np.zeros(shape=self.images[0], dtype=np.bool)
        for index, image in enumerate(self.images):
            tmp_img = np.expand_dims(image, 0)
            pred = np.argmax(model.predict(tmp_img))
            target = self.labels[index]
            if mode=='correct' and pred != target:
                to_keep[index] = True
            elif mode=='error' and pred == target:
                to_keep[index] = True
            else:
                raise ValueError(f'The key {mode} was not recognized')
        self.images= self.images[to_keep]
        self.labels = self.labels[to_keep]


class CompilerParameters(object):
    """
    The parameters that will be passed to the Keras model compiler.
    metrics can be either a single string ('acc') or a list of metrics from Keras.
    Additional metrics need to be imported.
    """

    def __init__(self):
        self._lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[15], values=[0.00005,0.00001])  #case 1: reduce to 0.000005 at 150
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._lr_schedule, amsgrad=True) # tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.01)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['acc']


class ModelGeometry(object):
    """A small object to store the variables needed to initialize a Keras NN"""
    def __init__(self, _pairmat):
        self.dimensions = _pairmat.pairwise_mats.shape[1:3]  # Get the image size
        self.input = np.append(self.dimensions, [1])  # Get the image size in 3D coordinates
        self.output = _pairmat.train.categorical_labels.shape[1]


class NNModel(object):
    """A base class for all Keras Model (technically a wrapper).
    Contains methods to compile and visualize models."""

    def __init__(self, model_geometry):
        if not isinstance(model_geometry, ModelGeometry):
            raise TypeError('Argument 1 must be a ModelGeometry object')
        input_layer = tf.keras.Input(shape=model_geometry.dimensions)

        outputs = Dense(units=model_geometry.output)(input_layer)
        self.NN = tf.keras.Model(input_layer, outputs)
        self.compiled = False
        self.generator = None
        self.test_generator = None
        self.history = None

    def compile_model(self, compiler_parameters):
        if not isinstance(compiler_parameters, CompilerParameters):
            raise TypeError('Argument 2 must be a CompilerParameters object')
        self.NN.compile(optimizer=compiler_parameters.optimizer,  # 'adam'
                        loss=compiler_parameters.loss,
                        metrics=compiler_parameters.metrics)
        self.compiled = True

    def plot(self, filename, show_shape=True):
        plot_model(self.NN, to_file='../Model/' + filename + '.png', show_shapes=show_shape)

    def train(self, pair_mat, training_parameters, test_pair_mat=None):
        """The training routine"""
        if not isinstance(pair_mat, PairMat):
            raise TypeError('Argument 2 must be a PairMat data object')
        if not self.compiled:
            raise ValueError('The model must be compiled before it can be trained')
        if not isinstance(training_parameters, TrainingParameters):
            raise TypeError('Argument 4 must be a TrainingParameters object')
        if test_pair_mat is not None and not isinstance(test_pair_mat, PairMat):
            raise ValueError('Argument 3 must be either None or a PairMat Data object.')

        logdir = '../Models/TensorBoard/' + datetime.now().strftime("MD%m%d_HMS%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, embeddings_freq=5, write_images=True)

        self.generator = tf.keras.preprocessing.image.ImageDataGenerator()
        self.np_generator = tf.keras.preprocessing.image.NumpyArrayIterator(x=pair_mat.pairwise_mats, 
                                                                            y=pair_mat.train.categorical_labels, 
                                                                            image_data_generator=self.generator, data_format='channels_last',
                                                                            batch_size=training_parameters.batch_size, shuffle=True)
        if test_pair_mat is None:
            self.history = \
                self.NN.fit(self.generator.flow(x=pair_mat.pairwise_mats,
                                                y=pair_mat.train.categorical_labels,
                                                batch_size=training_parameters.batch_size),
                            steps_per_epoch=pair_mat.pairwise_mats.shape[0] // training_parameters.batch_size,
                            verbose=2)
        else:
            # self.test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

            self.history = self.NN.fit(
                x=self.np_generator,
                #self.generator.flow(x=pair_mat.pairwise_mats,
                #                    y=pair_mat.train.categorical_labels,
                #                    batch_size=training_parameters.batch_size),
                steps_per_epoch=pair_mat.pairwise_mats.shape[0] // training_parameters.batch_size,
                epochs=training_parameters.epoch,
                validation_data=(test_pair_mat.pairwise_mats, test_pair_mat.train.categorical_labels),
                # batch_size=training_parameters.batch_size,
                #callbacks=[tensorboard_callback],
                validation_steps=test_pair_mat.pairwise_mats.shape[0] // training_parameters.batch_size, verbose=2)

    def save_model(self, model_name, prefix='../Models/'):
        label =  prefix + model_name + '.h5'
        self.NN.save(label)

    def import_model(self, model_name):
        """We want to import a pre-trained model and use it"""
        label =  '../Models/' + model_name + '.h5'
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
        line_4 = f'Randomize flag { str_state }\n'
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
        if not all(isinstance(models[x], NNModel) for x in range(len(models))):
            raise TypeError('At least one of the models input is not of the correct type')
        self.models = self.models.extend(models)
        self.length = + self.length + len(models)
        self.filled = True
        self.file_ = []

    def import_from_directory(self, target_dir):
        """
        Import all models in a target directory. Models will be added in the 
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
        #print(pred_returns, pred_returns.shape)
        #
        #for index in range(x.shape[0]):
        #    for model in range(len(self.models)):
        #        value = x[index, :, :].reshape(1, x.shape[1], x.shape[2], 1)
        #        pred_array[index, model] = np.argmax(self.models[model].predict(x=value))
        #    pred_returns[index] = np.argmax(np.bincount(pred_array[index, :].astype('int64')))
        if return_probabilities:
            # preds, scores
            return pred_array, pred_returns
        
        if return_vector:
            return pred_returns
        else:
            return pred_array
        # We want to evaluate each model at x and then assign 'points' based on each prediction,
        # Then return the prediction with the most points

    def evaluate(self, x_true, y_true, return_total=False):
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
            return np.sum(values)/len(values)

    def evaluate_on_each_model(self, x_true, y_true):
        assert self.filled, 'We must have a model to evaluate'
        y_pred = self.predict(x=x_true, return_vector=False)  # A matrix of predictions for each model
        # print('y_pred.shape is')
        print(y_pred.shape)
        y_model_acc = np.zeros(len(self.models))
        y_true = np.argmax(y_true, axis=1)
        for modelNo in range(0, self.length):
            y_model_acc[modelNo] = np.sum(np.equal(y_true, y_pred[:, modelNo]))/y_true.shape[0]

        return y_model_acc


class PairMat(object):
    # TODO Write documentation
    # TODO Write unit tests for PairMat
    """
    A class to store pairwise distance matrices including annotations
    and indices.
    """

    def __init__(self, file_name, num_classes=None, method='OLO', HC_method='ward'):
        """
        param: file_name: path-like object to data file
        param: num_classes: (optional) override and increase label creation size. 
        param: method (optional) string, which cluster method to use. Default is OLO
              availible options are 'None', 'HC' and 'OLO'.
        param: HC_method: string: valid method for 
        """
        
        self.vectors = None
        self.filepath =str(file_name)
        if os.path.isfile(self.filepath):
            self.valid = True
        else:
            message = 'The file path:' + self.filepath + ' is not valid'
            raise ValueError(message)

        # We have confirmed that the file is valid, now fill in the fields
        self.data = loadmat(self.filepath)
        self.pairwise_mats = np.zeros(shape=(self.data['matrices'].shape + (1,)))
        self.pairwise_mats[:, :, :, 0] = self.data['matrices']  # Input layer is expecting a samples x rows x col x channels
        self.labels = np.rint(self.data['indexes'][:, :2])  # skip lambda
        self.train = TrainingLabels(self, num_classes=num_classes)
        self.log_text = str(self.data['log'])
        self.NPop = self.data['NPop']
        self.R0_vals = self.data['values'][:,0]
        delattr(self, 'data')  # Remove the initial data object loaded in\\
        if method=='OLO':
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

    def append(self, newPairMat):
        raise NotImplementedError

    def leaf_order_optimize(self, method=hclust.ward):
        """
        Reorder the leafs of the images in place. Default single-linkage 
        ordering method computes a minimum spanning tree via nearest point algorithm.
        """
        for row in range(self.pairwise_mats.shape[0]):
            if row % 1000 == 0:
                print('Reached row',row)
            #    print(self.pairwise_mats[row,:,:,0])
            distvec = squareform(self.pairwise_mats[row,:,:,0])
            linkage_map = method(distvec)
            optimal_linkage_map = hclust.optimal_leaf_ordering(Z=linkage_map, y=distvec)
            order = hclust.leaves_list(optimal_linkage_map)
            self.pairwise_mats[row,:,:,0] = self.pairwise_mats[row, order, : ,0]
            self.pairwise_mats[row,:,:,0] = self.pairwise_mats[row,:,order,0]

    def hc(self, method='ward'):
        for row in range(self.pairwise_mats.shape[0]):
            if row % 1000 == 0:
                print('Reached row',row)
            distvec = squareform(self.pairwise_mats[row,:,:,0])
            linkage_map = hclust.linkage(distvec, method=method)
            order = hclust.leaves_list(linkage_map)
            self.pairwise_mats[row,:,:,0] = self.pairwise_mats[row, order, : ,0]
            self.pairwise_mats[row,:,:,0] = self.pairwise_mats[row,:,order,0]

    def show(self, image_id, cmap='viridis', ax=None, fig=None):
        if ax is None and fig is None:
            fig, ax = plt.subplots(1)
        image = np.squeeze(self.pairwise_mats[image_id,:,:,0])
        vmax = np.max(image)
        image = np.squeeze(image)
        main_IM = ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(main_IM, ax=ax)
        return fig, ax
            

class TrainingLabels(object):
    # TODO Write documentation for TrainingLabels
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
