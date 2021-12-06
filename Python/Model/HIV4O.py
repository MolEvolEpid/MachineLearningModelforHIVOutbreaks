import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
# Import the structures we need to build our model
from tensorflow.keras.models import Model


def HIV4O_functional(model_geometry):
    kernel_init = 'glorot_uniform'
    print(model_geometry.input)
    input_layer = keras.Input(shape=tuple(model_geometry.input))  # Pass in a tuple here
    x = Conv2D(32, (4,4), padding='same')(input_layer)
    x = Conv2D(32, (4, 4), kernel_initializer=kernel_init, name='Conv_1')(input_layer)
    x = BatchNormalization(name='BatchNorm_1')(x)   
    x = Conv2D(32, (4, 4), name='Conv_2')(x) 
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=kernel_init, name='Conv_3')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(name='BatchNorm_2')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=kernel_init, name='Conv_4')(x)
    if model_geometry.input[1] > 20:
        x = MaxPooling2D(pool_size=(2,2), name='MaxPool_2')(x)
    x = Flatten()(x)
    x = Dropout(rate=0.25, name='Dropout_0')(x)
    x = Dense(units=64, name='Dense_0', bias_regularizer=regularizers.l1(0.1), kernel_regularizer=regularizers.l2(0.05))(x) # on large models this is the largest layer
    x = Dense(units=64, name='Dense_1', bias_regularizer=regularizers.l1(0.01), kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(rate=0.25, name='Dropout_2')(x)
    output = Dense(units=model_geometry.output, activation='softmax', use_bias=False, name='Dense_output')(x)
    return Model(input_layer, output)
