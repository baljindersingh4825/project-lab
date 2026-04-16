import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []

for label, folder in enumerate(["closed", "open"]):
    path = os.path.join("dataset", folder)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (24,24))

        data.append(image)
        labels.append(label)

X = np.array(data).reshape(-1,24,24,1)/255.0
y = np.array(labels)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(24,24,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=10)

model.save("drowsiness_cnn.h5")

print("Model Saved!")