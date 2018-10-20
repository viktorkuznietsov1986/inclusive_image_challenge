from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Add, Activation, Flatten, \
    Dense, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU

def conv2d_block(x, filters, kernel_size, strides=1):
    """
    Implements the conv2d block with batch norm and leaky relu activation.
    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :return:
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding=('same' if strides==1 else 'valid'), kernel_regularizer=l2(5e-4))(x)
    return x

def conv2d_block_leaky(x, filters, kernel_size, strides=1):
    """
    Implements the conv2d block with batch norm and leaky relu activation.
    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :return:
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding=('same' if strides==1 else 'valid'), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def darknet_block(x, filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (0, 1)))(x)
    x = conv2d_block_leaky(x, filters, (3, 3), strides=2)

    for i in range(num_blocks):
        out = conv2d_block_leaky(x, filters // 2, kernel_size=(1, 1), strides=1)
        out = conv2d_block_leaky(out, filters, kernel_size=(3, 3), strides=1)

        x = Add()([x, out])

    return x


def darknet_body(x):
    x = conv2d_block_leaky(x, 32, 3)
    x = darknet_block(x, 64, 1)
    x = darknet_block(x, 128, 2)
    x = darknet_block(x, 256, 8)
    #x = darknet_block(x, 512, 8)
    #x = darknet_block(x, 1024, 4)

    return x

def darknet_classifier(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda img: (img/127.5)-1.)(inputs)
    x = darknet_body(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs, x)




