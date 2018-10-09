from keras import Input, Model
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Add, UpSampling2D, Concatenate, Activation
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
    x = darknet_block(x, 512, 8)
    x = darknet_block(x, 1024, 4)

    return x

def build_inference_layers(x, num_filters, out_filters):
    x = conv2d_block_leaky(x, num_filters, 1)
    x = conv2d_block_leaky(x, num_filters*2, 3)
    x = conv2d_block_leaky(x, num_filters, 1)
    x = conv2d_block_leaky(x, num_filters*2, 3)
    x = conv2d_block_leaky(x, num_filters, 1)

    y = conv2d_block_leaky(x, num_filters*2, 3)
    y = conv2d_block(y, out_filters, 1)

    return x, y

def yolo_v3(input_shape, num_anchors, num_classes):
    inputs = Input(shape=input_shape)
    x = darknet_body(inputs)
    darknet = Model(inputs, x)

    x, y1 = build_inference_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = conv2d_block_leaky(x, 256, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = build_inference_layers(x, 256, num_anchors*(num_classes+5))

    x = conv2d_block_leaky(x, 128, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = build_inference_layers(x, 128, num_anchors*(num_classes+5))

    model = Model(inputs, [y1, y2, y3])

    return model


model = yolo_v3((416,416,3), 9//3, 80)
model.load_weights('yolo.h5') # sanity check

#from keras.utils import plot_model
#plot_model(model, to_file='model1.png')


