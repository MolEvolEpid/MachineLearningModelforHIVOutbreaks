# TODO this needs a factory or an abstract class as an abstraction 
import tensorflow.keras as keras
from Structures import ModelGeometry, NNModel, CompilerParameters
from Model.HIV3O import HIV3O_functional
from Model.HIV1O import HIV1O_functional
from Model.HIV2O import HIV2O_functional
from Model.HIV12O import HIV12O_functional
from Model.HIV4O import HIV4O_functional


def check_NN_constructors(model_geometry, compiler_parameters):
    """A utility function to check that the variables passed to the NN constructors are the correct type"""
    if not isinstance(model_geometry, ModelGeometry):
        raise TypeError('Argument 1 must be a ModelGeometry object')
    if not isinstance(compiler_parameters, CompilerParameters):
        raise TypeError('Argument 2 must be a CompilerParameters object')
    return True


class HIV1O_c(NNModel):
    def __init__(self, model_geometry, compiler_parameters):
        if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
            super().__init__(model_geometry=model_geometry)
            self.NN = HIV1O_functional(model_geometry=model_geometry)
            self.compile_model(compiler_parameters=compiler_parameters)


class HIV2O_c(NNModel):
    def __init__(self, model_geometry, compiler_parameters):
        if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
            super().__init__(model_geometry=model_geometry)
            self.NN = HIV2O_functional(model_geometry=model_geometry)
            self.compile_model(compiler_parameters=compiler_parameters)


class HIV3O_c(NNModel):
    def __init__(self, model_geometry, compiler_parameters):
        if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
            super().__init__(model_geometry=model_geometry)
            self.NN = HIV3O_functional(model_geometry=model_geometry)
            self.compile_model(compiler_parameters=compiler_parameters)


class HIV12O_c(NNModel):
    def __init__(self, model_geometry, compiler_parameters):
        if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
            super().__init__(model_geometry=model_geometry)
            self.NN = HIV12O_functional(model_geometry=model_geometry)
            self.compile_model(compiler_parameters=compiler_parameters)


class HIV4O_c(NNModel):
    def __init__(self, model_geometry, compiler_parameters):
        if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
            super().__init__(model_geometry=model_geometry)
            self.NN = HIV4O_functional(model_geometry=model_geometry)
            self.compile_model(compiler_parameters=compiler_parameters)



# To make a new model, add the function in a file within the Model subpackage, import the module with the package,
# then create a new model as:
#
# class HIV12O(NNModel):
#     def __init__(self, model_geometry, compiler_parameters):
#         if check_NN_constructors(model_geometry=model_geometry, compiler_parameters=compiler_parameters):
#             super().__init__(model_geometry=model_geometry)
#             self.NN = HIV12O_functional(model_geometry=model_geometry)
#             self.compile_model(compiler_parameters=compiler_parameters)
