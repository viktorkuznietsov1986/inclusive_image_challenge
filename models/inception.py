from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications


def build_inceptionv3_based_classifier(input_shape, num_classes):
    inceptionv3 = applications.InceptionV3(include_top=False,
                                    input_shape=input_shape)
    inceptionv3.trainable = False

    model = Sequential()
    model.add(inceptionv3)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                        activation='sigmoid'))

    return model