from keras.models import Model
from keras.layers import Conv2D, Activation, BatchNormalization, concatenate
from keras.layers import Input, MaxPool2D, UpSampling2D

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.vis_utils import plot_model

import keras.backend as K

def fire_module(x, fire_id, squeeze=16, expand=64):
    """
    The function implements fire module of improved SqueezeNet
    with a residual connection over the exand layers. The module
    also uses batch normalization.
    """

    # Some naming helpers
    s_id = 'fire' + str(fire_id) + '/'
    sq1x1 = 'squeeze1x1'
    exp1x1 = 'expand1x1'
    exp3x3 = 'expand3x3'

    # We support here any channel format (TensorFlow or Theano)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    
    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + 'relu_' + sq1x1)(x)

    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + 'relu_' + exp1x1)(left)

    right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + 'relu_' + exp3x3)(right)

    # Add residual connection over the expansion layer
    x = concatenate([left, right, x], axis=channel_axis, name=s_id + 'concat')

    x = BatchNormalization(axis=channel_axis)(x)

    return x

def build_model(input_tensor=None, input_shape=None):
    # Verify requested input shape
    input_shape = _obtain_input_shape(input_shape, default_size=256, min_size=48,
                                      data_format=K.image_data_format(), require_flatten=False)

    # Model takes into account predecessor layers
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
     
    # Encoder
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    
    pool1 = x
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    
    pool3 = x
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)

    pool5 = x
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(x)
    
    # High-level feature processing
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    # Decoder
    x = UpSampling2D()(x)
    x = concatenate([x, pool5], axis=channel_axis, name='pool5_concat')
    x = Conv2D(256, (3, 3), padding='same', name='pool5_conv')(x)
    x = Activation('relu', name='pool5_relu')(x)

    x = BatchNormalization(axis=channel_axis)(x)

    x = UpSampling2D()(x)
    x = concatenate([x, pool3], axis=channel_axis, name='pool3_concat')
    x = Conv2D(128, (3, 3), padding='same', name='pool3_conv')(x)
    x = Activation('relu', name='pool3_relu')(x)

    x = BatchNormalization(axis=channel_axis)(x)

    x = UpSampling2D()(x)
    x = concatenate([x, pool1], axis=channel_axis, name='pool1_concat')
    x = Conv2D(64, (3, 3), padding='same', name='pool1_conv')(x)
    x = Activation('relu', name='pool1_relu')(x)

    x = BatchNormalization(axis=channel_axis)(x)

    # Output mapping
    x = UpSampling2D()(x)
    x = Conv2D(3, (3, 3), padding='same', name='output_conv')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(inputs, x, name='squeezenet')

if __name__ == '__main__':
    model = build_model(input_shape=(256, 256, 3))
    
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)