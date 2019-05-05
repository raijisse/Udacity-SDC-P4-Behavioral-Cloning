#title           :model.py
#description     :This will create a header for a python script.
#author          :regis_rauzy@live.com
#date            :20190505
#usage           :python model.py
#notes           :
#python_version  : 3.6.8
#==============================================================================

import os
import csv
from math import ceil
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import ndimage

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Conv2D, Dropout


# First, we are going to list all samples that we have in the data set
# They are included in the driving_log.csv stored in the /data folder
# Note that, this folder is not committed so you have to change the path to your
# own data

samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    next(reader, None) # Skip header
    for line in reader:
        samples.append(line)

# We then randomly generate a training sample and a validation sample
train_samples, validation_samples = train_test_split(samples, test_size=0.25)

# Method to feed generator to our Keras model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    # Read in original image (i=0 -> center, i=1 -> left, i=2 -> right)
                    image = ndimage.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    # If center image (i=0):
                    if i == 0:
                        angles.append(angle)
                    # If image from left (i=1), correct with positive bias
                    elif i == 1:
                        angles.append(angle + 0.2)
                    # If image from right (i=2), correct with negative bias
                    elif i==2:
                        angles.append(angle - 0.2)

                    # Then we augment the data set by flipping the image:
                    images.append(cv2.flip(image, 1))
                    # And saving the flipped angle:
                    if i == 0:
                        angles.append(angle*-1.0)
                    # If image from left (i=1), correct with positive bias
                    elif i == 1:
                        angles.append((angle + 0.2)*-1.0)
                    # If image from right (i=2), correct with negative bias
                    elif i==2:
                        angles.append((angle - 0.2)*-1.0)

            # Convert our image and steering measurements to array to be Keras compatible
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# Create generator that will be feed to Keras
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Define our model architecture, here we copied the Nvidia architecture
# and modified it slightly by adding dropout in the fully conntected layers

def nvidia_architecture():
    model_ = Sequential()
    model_.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model_.add(Cropping2D(cropping=((70,25), (0,0))))
    model_.add(Conv2D(24,(5,5), strides = (2,2), activation = 'relu'))
    model_.add(Conv2D(36,(5,5), strides = (2,2), activation = 'relu'))
    model_.add(Conv2D(48,(5,5), strides = (2,2), activation = 'relu'))
    model_.add(Conv2D(64,(3,3), activation = 'relu'))
    model_.add(Conv2D(64,(3,3), activation = 'relu'))
    model_.add(Flatten())
    model_.add(Dropout(0.25))
    model_.add(Dense(100, activation='relu'))
    model_.add(Dropout(0.25))
    model_.add(Dense(50, activation='relu'))
    model_.add(Dropout(0.25))
    model_.add(Dense(10, activation='relu'))
    model_.add(Dense(1))
    return model_


model = nvidia_architecture()
print(model.summary())

# Then, we compile and train our model using an adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=7, verbose=1)

# Saving the model
model.save('model.h5')

print("You should now be able to use the model to drive on the test track ! :D")
