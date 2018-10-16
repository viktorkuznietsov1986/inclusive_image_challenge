import cv2
import os

train_raw_dir = '../train_data/'
train_processed_dir = '../train_data/scaled/'

dim = (500, 500)

raw_files = os.listdir(train_raw_dir)

for index, file in enumerate(raw_files):
    #if index > 10:
    #    break

    file_name = train_raw_dir + file
    image = cv2.imread(file_name)
    resized = cv2.resize(image, dim)

    out_file_name = train_processed_dir + file
    cv2.imwrite(out_file_name, resized)


