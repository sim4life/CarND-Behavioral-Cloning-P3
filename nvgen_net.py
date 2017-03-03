import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # to skip header row in original data csv
    for line in csv_reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


correction      = 0.2 # this is a parameter to tune
top_crop        = 70
bot_crop        = 25
img_path        = './data/IMG/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augmented_images, augmented_angles = [], []
            for batch_sample in batch_samples:
                file_name = batch_sample[0].split('/')[-1] # centre img
                images.append(cv2.imread(img_path + file_name))
                file_name = batch_sample[1].split('/')[-1] # left img
                images.append(cv2.imread(img_path + file_name))
                file_name = batch_sample[2].split('/')[-1] # right img
                images.append(cv2.imread(img_path + file_name))

                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                angles.extend([center_angle, left_angle, right_angle])

            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# row, col, ch = 160 - top_crop - bot_crop, 320, 3  # Trimmed image format
ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((top_crop,bot_crop),(0,0)), input_shape=(ch,row,col)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
            validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model_nvgen.h5')
