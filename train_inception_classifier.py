import os

import sklearn
import pandas as pd
import numpy as np
import cv2
import pickle

import math

import tensorflow as tf
import keras.backend as K


from models.inception import build_inceptionv3_based_classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

classes_file_name = './data/classes-trainable.csv'
classes_file = pd.read_csv(classes_file_name)

all_labels = classes_file['label_code']

# set the number of labels which will be used as an output layer size for a model
num_labels = len(all_labels)

# build the index dictionary based on the labels collection
labels_index = {label:idx for idx, label in enumerate(all_labels)}

# retrieve the list of labels assigned to all images
labels_set = set(all_labels)

images_with_labels = {}

# set up the threshold for the confidence of the machine label
machine_label_threshold = .4

train_images_dir = '../train_data/'
scaled_train_images = os.listdir(train_images_dir)

print (len(scaled_train_images))


def load_data():
    with open('./data/train.pkl', 'rb') as handle:
        data = pickle.load(handle)

    return data


data = load_data()
print (len(data))

train_images = []

for train_image in scaled_train_images:
    key = train_image[:-4]

    if key in data:
        train_images.append(train_image)

print (len(train_images))

# split the samples to train and dev sets
# shuffle first
sklearn.utils.shuffle(train_images)

# the split is based on the current number of images - about 350000. there should be about 10500 images for dev set
train_samples, validation_samples = train_test_split(train_images, test_size=0.03)


# do the multi-hot encoding
def multi_hot_encode(x, num_classes):
    labels_encoded = np.zeros(num_classes)

    labels_encoded[x] = 1

    return labels_encoded


input_shape = (450, 450, 3)


def normalize(img):
    return (img / 127.5) - 1.


def generate_occurences(data):
    occurences = {}

    for value in data.values():
        for label in value:
            if label in occurences:
                occurences[label] += 1
            else:
                occurences[label] = 1

    return occurences

labels_dict = generate_occurences(data)

def create_class_weight(labels_dict, mu=0.15):
    # labels_dict : {ind_label: count_label}
    # mu : parameter to tune

    total = sum(labels_dict.values())
    print (total)
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

W = create_class_weight(labels_dict)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# define the generator method which loads images in a batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            labels = []

            for batch_sample in batch_samples:
                image_name = train_images_dir + batch_sample
                image = cv2.imread(image_name)
                image = cv2.resize(image, (input_shape[0], input_shape[1]))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = normalize(image)
                key = batch_sample[:-4]

                #if key in data:
                label = multi_hot_encode(data[key], num_labels)
                #else:
                #    label = np.zeros(num_labels)

                images.append(image)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)


model = build_inceptionv3_based_classifier(input_shape, num_labels)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

"""
model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4,
                                           momentum=0.9),
                  metrics=['accuracy'])
"""

model.summary()

num_epochs = 5

batch_size = 8

from keras.callbacks import ModelCheckpoint, EarlyStopping

# trains the model
# defined 2 callbacks: early stopping and checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, epochs=3):
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1) # no need to do early stopping as the model needs baby-sitting
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_generator, class_weight=W, steps_per_epoch=len(train_samples)//batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=epochs, callbacks=[checkpoint_callback], )


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

train_model(model, train_generator, validation_generator, num_epochs)