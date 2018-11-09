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

test_images_dir = './data/stage_2_test_images/'
test_images = os.listdir(test_images_dir)

print (test_images[0])

input_shape = (450, 450, 3)

model = build_inceptionv3_based_classifier(input_shape, num_labels)
model.load_weights('model.h5')
model.summary()

n_channels = 3


def normalize(img):
    return (img / 127.5) - 1.


def predict(model, image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize(image)
    image = np.reshape(image, [1, input_shape[0], input_shape[1], n_channels])
    prediction = model.predict(image)[0]

    indices02 = np.argwhere(prediction >= 0.2).flatten()
    labels02 = " ".join(all_labels[indices02].values)

    indices03 = np.argwhere(prediction >= 0.3).flatten()
    labels03 = " ".join(all_labels[indices03].values)

    indices04 = np.argwhere(prediction >= 0.4).flatten()
    labels04 = " ".join(all_labels[indices04].values)

    indices05 = np.argwhere(prediction >= 0.5).flatten()
    labels05 = " ".join(all_labels[indices05].values)

    return labels02, labels03, labels04, labels05


submission02 = {'image_id': [], 'labels': []}
submission03 = {'image_id': [], 'labels': []}
submission04 = {'image_id': [], 'labels': []}
submission05 = {'image_id': [], 'labels': []}

for index, img in enumerate(test_images):
    image_name = test_images_dir + img

    if index % 1000 == 0:
        print("processed {} images".format(index))

    image_id = img[:-4]

    labels02, labels03, labels04, labels05 = predict(model, image_name)

    submission02['image_id'].append(image_id)
    submission02['labels'].append(labels02)

    submission03['image_id'].append(image_id)
    submission03['labels'].append(labels03)

    submission04['image_id'].append(image_id)
    submission04['labels'].append(labels04)

    submission05['image_id'].append(image_id)
    submission05['labels'].append(labels05)

submission02 = pd.DataFrame(submission02)
submission02.to_csv('stage2_inception_450x450_02.csv', index=False)

submission03 = pd.DataFrame(submission03)
submission03.to_csv('stage2_inception_450x450_03.csv', index=False)

submission04 = pd.DataFrame(submission04)
submission04.to_csv('stage2_inception_450x450_04.csv', index=False)

submission05 = pd.DataFrame(submission05)
submission05.to_csv('stage2_inception_450x450_05.csv', index=False)