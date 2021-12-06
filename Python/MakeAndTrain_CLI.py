import argparse
import os
import numpy as np
from Models import HIV4O_c as model #pylint: disable=no-name-in-module
from Structures import ModelGeometry, PairMat, CompilerParameters, NNModel, TrainingParameters #pylint: disable=import-error
from datetime import datetime

# First we define an argument parser so we can use this as the command line utility
parser = argparse.ArgumentParser(description='Accepts paths to validation and training data')

# The first argument of the argparser should be a path to the validation data
parser.add_argument('--test', required=True, dest='testing_list')

# Accept a list of training files
parser.add_argument('--train', required=True, nargs='+', dest='training_list')

# Specify the choice of ordering to apply
parser.add_argument('--ordering', required=True, dest='ordering', choices=['OLO', 'HC', 'None'])

# Now parse the args into the namespace
args = parser.parse_args()

datums = [PairMat(data, method=args.ordering) for data in args.training_list]  # Import the training data
nSamples = datums[0].pairwise_mats.shape[1]
test_data = PairMat(args.testing_list, method=args.ordering)  # Import the test data
geo = ModelGeometry(datums[0])   # pass in a
print('Begin making models')
compiler_params = CompilerParameters()  # we might not want the default options, set them now
compiler_params.loss = 'categorical_crossentropy'
compiler_params.metrics = ['acc', 'categorical_crossentropy']

# Print these to console for logging purposes
print('dimension: ',geo.dimensions)
print('output: ', geo.output)
print('input: ', geo.input)

models = [model(model_geometry=geo, compiler_parameters=compiler_params) for data in datums]
models[0].NN.summary()
# initialize models with correct geometry
# Set training parameters
training_parameters = TrainingParameters()
training_parameters.epoch = 50 
training_parameters.batch_size = 32 



print('Reach end of setup and allocations. Proceeding to train models')
the_time = datetime.now()
dt_string = the_time.strftime("%d-%m-%Y--%H-%M-%S")
print(dt_string)
for index in range(len(args.training_list)):
    models[index].train(pair_mat=datums[index],
                        training_parameters=training_parameters,
                        test_pair_mat=test_data)

    print('completed training model from: ' + str(index))
    model_dir_name = 'HIV_1x3-prod-'+str(nSamples)+'/'+str(args.ordering)  # save directory hard-coded here
    try:
        os.makedirs('~' + os.path.realpath(os.path.dirname(__file__) + '/../Model/'+model_dir_name+'/'))
    except OSError: # dir already exists - notify user and continue
        print('~' + os.path.realpath(os.path.dirname(__file__) + '/../Model/'+model_dir_name+'/'))
        print('Caught OSError, dir already exists')

    filename = '/P-' + str(index) + '-time-' + dt_string
    print(filename)  # The actual model file name
    models[index].save_model(filename, prefix=f'./Models_tmp/HIV_1x3-realmodel-/Order-{args.ordering}-{nSamples}/')
    
    #  Let's run out some per-label statistics
    pred_vector = models[index].predict(x=test_data.pairwise_mats)
    pred_vals = np.argmax(pred_vector, axis=1)
    for label in range(np.amax(test_data.train.labels)):
        # use a boolean mask to find the index of images with the desired labels
        inds = np.where(np.argmax(test_data.train.categorical_labels, axis=1) == label, True, False)
        print('The selected index shape for label', label, ' is ', inds.shape)  # For verification - how

        print('The predictions matrix has shape:', pred_vals.shape)
        acc = np.sum(pred_vals[inds] == np.argmax(test_data.train.categorical_labels, axis=1)[inds])
        print('Label: 'label, '\tAccuracy:', acc)

print('Model training and analysis complete. Exiting script.')

