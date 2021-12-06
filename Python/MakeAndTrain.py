print('hello world')
import os
from Models import HIV4O_c as model #pylint: disable=no-name-in-module
from Structures import ModelGeometry, PairMat, CompilerParameters, NNModel, TrainingParameters #pylint: disable=import-error
from datetime import datetime

training_list = ['../Data/MultiModelTrainingData/HIV_1x3/HIV_1i/Train30.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/HIV_2i/Train30.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/HIV_3i/Train30.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/HIV_4i/Train30.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/HIV_5i/Train30.mat']

testing_list = ['../Data/MultiModelTrainingData/HIV_1x3/HIV_1i/Test30.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/HIV_2i/Test30.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/HIV_3i/Test30.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/HIV_4i/Test30.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/HIV_5i/Test30.mat']

"""
training_list = ['../Data/MultiModelTrainingData/HIV_1x3/Experimental/Train.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Train.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Train.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Train.mat',
                 '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Train.mat']

testing_list = ['../Data/MultiModelTrainingData/HIV_1x3/Experimental/Test.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Test.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Test.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Test.mat',
                '../Data/MultiModelTrainingData/HIV_1x3/Experimental/Test.mat']
"""
print(os.getcwd())
print(os.path.dirname(__file__))
datums = [PairMat(data) for data in training_list]  # Make data files

test_data = [PairMat(test) for test in testing_list]  # Make test files
geo = ModelGeometry(datums[0])
print('Begin making models')
compiler_params = CompilerParameters()
compiler_params.loss = 'categorical_crossentropy'  # kullback_leibler_divergence
compiler_params.metrics = ['acc', 'categorical_crossentropy']
print('dimension: ',geo.dimensions)
print('output: ', geo.output)
print('input: ', geo.input)

models = [model(model_geometry=geo, compiler_parameters=compiler_params) for data in datums]
# initialize models with correct geometry
# Set training parameters
training_parameters = TrainingParameters()
training_parameters.epoch = 300  # For convenience
training_parameters.batch_size = 64  # for convenience
print('Reach end of setup and allocations. Proceeding to train models')
the_time = datetime.now()
dt_string = the_time.strftime("%d-%m-%Y--%H-%M-%S")
print(dt_string)
for index in range(len(training_list)):
    # models[index].plot('HIV' + str(index) + '-time' + dt_string + '-num-' + str(index + 1))
    models[index].train(pair_mat=datums[index],
                        training_parameters=training_parameters,
                        test_pair_mat=test_data[index])

    print('completed training model from: ' + str(index))
    model_dir_name = 'HIV_1x3-tf2-s-30'
    try:
        os.mkdir('~' + os.path.realpath(os.path.dirname(__file__) + '/../Models/'+model_dir_name))
    except OSError:
        pass  # dir already exists
    filename = model_dir_name + '/' + str(index) + '-time-' + dt_string + '-num-' + str(index + 1)
    print(filename)
    models[index].save_model(filename)
