from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras import applications


def build_inceptionv3_based_classifier(input_shape, num_classes):
    inceptionv3 = applications.InceptionV3(weights=None, include_top=False,
                                    input_shape=input_shape)
    inceptionv3.trainable = True

    model = Sequential()
    model.add(inceptionv3)
    model.add(MaxPooling2D(pool_size=2), strides=2)
    model.add(Flatten())
    model.add(Dense(num_classes*3//2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='sigmoid'))

    return model