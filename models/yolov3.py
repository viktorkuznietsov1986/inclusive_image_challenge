from keras import Input, Model
from keras.layers import UpSampling2D, Concatenate

from models.darknet53 import darknet_body, conv2d_block_leaky, conv2d_block


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