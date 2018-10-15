import sklearn

from models.darknet53 import darknet_classifier

labels_file_name = './data/train_machine_labels.csv'

# split the samples to train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.02)

# define the generator method which loads images in a batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_name = data_folder_name + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


num_classes = 7178
input_shape = (500, 500, 3)

model = darknet_classifier(input_shape, num_classes)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()