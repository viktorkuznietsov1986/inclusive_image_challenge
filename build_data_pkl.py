import os
import pickle
import pandas as pd

labels_file_name = '../train_data/train_machine_labels.csv'
labels_file = pd.read_csv(labels_file_name)

human_file_name = '../train_data/train_human_labels.csv'
human_labels_file = pd.read_csv(human_file_name)

classes_file_name = '../train_data/classes-trainable.csv'
classes_file = pd.read_csv(classes_file_name)

all_labels = classes_file['label_code']

# set the number of labels which will be used as an output layer size for a model
num_labels = len(all_labels)

# build the index dictionary based on the labels collection
labels_index = {label: idx for idx, label in enumerate(all_labels)}

# retrieve the list of labels assigned to all images
labels_set = set(all_labels)

images_with_labels = {}

# set up the threshold for the confidence of the machine label
machine_label_threshold = .1

#processed_labels = labels_file.loc[(labels_file['Confidence'] > machine_label_threshold), ['ImageID', 'LabelName']]

#labels_file = processed_labels
#print (labels_file.head())

train_images_dir = '../train_data/scaled/'
scaled_train_images = os.listdir(train_images_dir)


def prepare_data():
    images_dict = {}

    for index, image_file_name in enumerate(scaled_train_images):
        if index > 10:
            break

        image_file_name_wo_ext = image_file_name[:-4]
        print(image_file_name_wo_ext)
        lst = labels_file.loc[labels_file['ImageID'] == image_file_name_wo_ext][
            'LabelName'].tolist()  # [labels_file['Confidence'] > machine_label_threshold]
        integers_lst = []
        for item in lst:
            if item in labels_index:
                integers_lst.append(labels_index[item])
        images_dict[image_file_name_wo_ext] = integers_lst

    return images_dict

def prepare_labels_for_full_dataset(data):
    images_dict = {}

    for i, row in data.iterrows():
        image_id = row['ImageID']
        label = row['LabelName']

        print (image_id)

        if label in labels_index:
            item = labels_index[label]

            if image_id in images_dict:
                images_dict[image_id].append(item)
            else:
                images_dict[image_id] = [item]

    return images_dict

def prepare_human_labels_for_full_dataset(data):
    images_dict = {}

    for i, row in data.iterrows():
        image_id = row['ImageID']
        label = row['LabelName']

        print (image_id)

        if label in labels_index:
            item = labels_index[label]

            if image_id in images_dict:
                images_dict[image_id].append(item)
            else:
                images_dict[image_id] = [item]

    return images_dict


def load_data():
    with open('./data/train.pkl', 'rb') as handle:
        data = pickle.load(handle)

    return data

def load_human_data():
    with open('./data/train_human.pkl', 'rb') as handle:
        data = pickle.load(handle)

    return data

#data = load_data()

#data = prepare_data()
#data = prepare_labels_for_full_dataset(processed_labels)

#human_data = prepare_human_labels_for_full_dataset(human_labels_file)


#print (human_data)

def save_human_data(data):
    with open("../train_data/train_human.pkl", "wb") as f:
        pickle.dump(data, f)

machine_data = load_data()
human_data = load_human_data()

print (len(machine_data))

def get_intersection(machine_data, human_data):
    for human_key in human_data.keys():
        machine_data[human_key] = human_data[human_key]

    return machine_data

intersection = get_intersection(machine_data, human_data)

print (len(intersection))


save_human_data(intersection)

print (human_data['000002b66c9c498e'])