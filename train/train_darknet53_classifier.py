from models.darknet53 import darknet_classifier

num_classes = 7178
input_shape = (416, 416, 3)

model = darknet_classifier(input_shape, num_classes)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()

