import argparse
import os
import numpy as np
from Models import HIV4O_c as model  # pylint: disable=no-name-in-module
from Structures import ModelGeometry, PairMat, CompilerParameters, NNModel, TrainingParameters  # pylint: disable=import-error
from datetime import datetime

# First we define an argument parser so we can use this as the command line utility
parser = argparse.ArgumentParser(description='Accepts paths to validation and training data')

# The first argument of the argparser should be a path to the validation data
parser.add_argument('--test', required=True, dest='testing_list')

# Accept a list of training files
parser.add_argument('--train', required=True, nargs='+', dest='training_list')

# Specify the choice of ordering to apply
parser.add_argument('--ordering', required=True, dest='ordering', choices=['OLO', 'HC', 'None'])

# Optional extra keyword for specifying save directory
parser.add_argument('--savekey', required=False, dest='savekey', default='.', type=str)

# Now parse the args into the namespace
args = parser.parse_args()
print(args)
datums = [PairMat(data, method=args.ordering) for data in args.training_list]  # Import the training data
nSamples = datums[0].pairwise_mats.shape[1]
test_data = PairMat(args.testing_list, method=args.ordering)  # Import the test data
geo = ModelGeometry(datums[0])  # pass in a
# Print these to console for logging purposes
print('dimension: ', geo.dimensions)
print('output: ', geo.output)
print('input: ', geo.input)
print('Begin making models')
compiler_params = CompilerParameters(loss='categorical_crossentropy',
                                     metrics=['acc', 'categorical_crossentropy'])
# we might not want the default options


models = [model(model_geometry=geo, compiler_parameters=compiler_params) for data in datums]
models[0].NN.summary()
# initialize models with correct geometry

# Set training parameters
training_parameters = TrainingParameters(epoch=100, batch_size=32)

print('Reach end of setup and allocations. Proceeding to train models')
the_time = datetime.now()
dt_string = the_time.strftime("%d-%m-%Y--%H-%M-%S")
print(dt_string)
for index in range(len(args.training_list)):
    models[index].train(pair_mat=datums[index],
                        training_parameters=training_parameters,
                        test_pair_mat=test_data)

    print(f'completed training model from: {index}')
    model_dir_name = f'User_models/{args.savekey}/Model-{nSamples}/{args.ordering}'  # save directory hard-coded here
    model_dir = os.path.abspath(os.sep) + os.path.realpath(os.path.dirname(__file__) + '/../Trained_models/' + model_dir_name + '/')
    try:
        os.makedirs(model_dir)
    except OSError:  # dir already exists - notify user and continue
        print(model_dir)
        print('Caught OSError, dir already exists')

    filename = '/P-' + str(index) + '-time-' + dt_string
    print(filename)  # The actual model file name
    # "Models_tmp" is the default save directory for trained models
    models[index].save_model(filename, prefix=f'../Trained_models/{model_dir_name}/Order-{args.ordering}-{nSamples}/')

    #  Let's run out some per-label statistics
    pred_vector = models[index].predict(x=test_data.pairwise_mats)
    pred_vals = np.argmax(pred_vector, axis=1)
    for label in range(np.amax(test_data.train.labels) - 1):
        # use a boolean mask to find the index of images with the desired labels
        inds = np.where(np.argmax(test_data.train.categorical_labels, axis=1) == label, True, False)
        print(f'The selected index shape for label {label} is {inds.shape}')
        print('The predictions matrix has shape:', pred_vals.shape)
        acc = np.sum(pred_vals[inds] == np.argmax(test_data.train.categorical_labels, axis=1)[inds])
        print(f'Label: {label} \tAccuracy: {acc}')

print('Model training and analysis complete. Exiting script.')
