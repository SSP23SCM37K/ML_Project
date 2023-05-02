'''
ML project
Driver drowsiness detection

Training ML model

By
Sree Rama Murthy Kandala
Aasritha Juvva
Pavan Kumar Turpati
'''
import os
import random
import shutil
import scipy
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

def generator(directory, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(directory, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

batch_size = 32
target_size = (24, 24)

train_generator = generator('dataset_new/train', shuffle=True, batch_size=batch_size, target_size=target_size)
test_generator = generator('dataset_new/test', shuffle=True, batch_size=batch_size, target_size=target_size)

num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(train_generator.classes) // batch_size
validation_steps = len(test_generator.classes) // batch_size

model.fit(train_generator, validation_data=test_generator, epochs=15, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

model.save('cnnmodel.h5', overwrite=True)
