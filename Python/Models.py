"""
Models.py
Michael Kupperman

Define classes here for models. Can extend the framework by importing additional models.
"""
from Model.HIV4O import HIV4O_functional
from Structures import ModelGeometry, NNModel, CompilerParameters


def check_NN_constructors(model_geometry, compiler_parameters):
    """A utility function to check that the variables passed to the NN constructors are the correct type"""
    if not isinstance(model_geometry, ModelGeometry):
        raise TypeError('Argument 1 must be a ModelGeometry object')
    if not isinstance(compiler_parameters, CompilerParameters):
        raise TypeError('Argument 2 must be a CompilerParameters object')
    return True


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
