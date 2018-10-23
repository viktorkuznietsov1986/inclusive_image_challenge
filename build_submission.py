import os
import pandas as pd
import numpy as np
import cv2

from models.inception import build_inceptionv3_based_classifier

classes_file_name = './data/classes-trainable.csv'
classes_file = pd.read_csv(classes_file_name)
all_labels = classes_file['label_code']

# set the number of labels which will be used as an output layer size for a model
num_labels = len(all_labels)

# build the index dictionary based on the labels collection
labels_index = {label:idx for idx, label in enumerate(all_labels)}

test_images_dir = './data/stage_1_test_images/'
test_images = os.listdir(test_images_dir)

print (test_images[0])

input_shape = (450, 450, 3)

model = build_inceptionv3_based_classifier(input_shape, num_labels)
model.load_weights('model.h5')
model.summary()

n_channels = 3


def normalize(img):
    return (img / 127.5) - 1.


def predict(model, image_name, threshold=0.1):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = normalize(image)
    image = np.reshape(image, [1, input_shape[0], input_shape[1], n_channels])
    prediction = model.predict(image)[0]

    indices = np.argwhere(prediction >= threshold).flatten()
    labels = " ".join(all_labels[indices].values)

    return labels


submission = {'image_id': [], 'labels': []}

for index, img in enumerate(test_images):
    image_name = test_images_dir + img

    if index % 1000 == 0:
        print("processed {} images".format(index))

    image_id = img[:-4]

    labels = predict(model, image_name)

    submission['image_id'].append(image_id)
    submission['labels'].append(labels)


submission = pd.DataFrame(submission)
submission.to_csv('inception_450x450_1_epoch.csv', index=False)