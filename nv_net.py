import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./data/driving_log.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # to skip header row in original data csv
    for line in csv_reader:
        lines.append(line)


correction      = 0.2 # this is a parameter to tune
top_crop        = 70
bot_crop        = 25
img_path        = './data/IMG/'
images          = []
measurements    = []
for line in lines:
    file_name = line[0].split('/')[-1] # centre img
    images.append(cv2.imread(img_path + file_name))
    file_name = line[1].split('/')[-1] # left img
    images.append(cv2.imread(img_path + file_name))
    file_name = line[2].split('/')[-1] # right img
    images.append(cv2.imread(img_path + file_name))

    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    measurements.extend([steering_center, steering_left, steering_right])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

row, col, ch = 160, 320, 3  # Raw image format
# row, col, ch = 160 - top_crop - bot_crop, 320, 3  # Trimmed image format

model = Sequential()
# model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(row,col,ch)))
# model.add(Cropping2D(cropping=((top_crop,bot_crop),(0,0)), input_shape=(row,col,ch)))
model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(ch,row,col), output_shape=(ch,row,col)))
# model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(row,col,ch), output_shape=(row,col,ch)))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model_nv.h5')
