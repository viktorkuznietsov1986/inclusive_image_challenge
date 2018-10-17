import os

import sklearn
import pandas as pd
import numpy as np
import cv2
import pickle

from models.darknet53 import darknet_classifier

#labels_file_name = './data/train_machine_labels.csv'
#labels_file = pd.read_csv(labels_file_name)

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
machine_label_threshold = .5

train_images_dir = '../train_data/scaled/'
scaled_train_images = os.listdir(train_images_dir)

print (len(scaled_train_images))


def load_data():
    with open('./data/train.pkl', 'rb') as handle:
        data = pickle.load(handle)

    return data


data = load_data()

# split the samples to train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(scaled_train_images, test_size=0.01)


# do the multi-hot encoding
def multi_hot_encode(x, num_classes):
    labels_encoded = np.zeros(num_classes)

    for label in x:
        if label in labels_index:
            label = labels_index[label]
            labels_encoded[label] = 1

    return labels_encoded

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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (416, 416))
                key = batch_sample[:-4]
                label = multi_hot_encode(data[key], num_labels)

                images.append(image)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)


input_shape = (416, 416, 3)

model = darknet_classifier(input_shape, num_labels)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()

num_epochs = 5

batch_size = 32

from keras.callbacks import ModelCheckpoint, EarlyStopping

# trains the model
# defined 2 callbacks: early stopping and checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, epochs=3):
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=epochs, callbacks=[early_stopping_callback, checkpoint_callback], )


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

train_model(model, train_generator, validation_generator, num_epochs)