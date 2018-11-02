from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from keras import applications


def build_inceptionv3_based_classifier(input_shape, num_classes):
    inceptionv3 = applications.InceptionV3(weights=None, include_top=False,
                                    input_shape=input_shape)
    inceptionv3.trainable = True

    model = Sequential()
    model.add(inceptionv3)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='sigmoid'))

    return model