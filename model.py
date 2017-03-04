import csv
import cv2
import os
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', './data', "The path to dataset.")

def get_immediate_subdirs(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

# Loading data samples from CLI passed arg data_path
def load_data_samples():
    log_file_paths = []
    file_paths = get_immediate_subdirs(FLAGS.data_path)
    if len(file_paths) > 1:
        for p in file_paths:
            log_file_paths.append(FLAGS.data_path+'/'+p)
    else:
        log_file_paths = [FLAGS.data_path]

    print("data files are: =", log_file_paths)
    samples = []
    # loading from all csv files present within the CLI passed arg data_path
    for log in log_file_paths:
        print("Reading ={}".format(log+'/driving_log.csv'))
        with open(log+'/driving_log.csv') as csv_file:
            csv_reader = csv.reader(csv_file)
            line1 = next(csv_reader) # to skip header row in original data csv
            if line1[0].find('/') is not -1:
                samples.append(line1) # don't ignore if a valid row
            for line in csv_reader:
                samples.append(line)

            print("sample size is: {}".format(len(samples)))

    # splitting data into training samples and validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

# Getting the sub_path of full path based on CLI passed arg data_path
def get_last_half_path(full_path):
    last_dir = FLAGS.data_path.strip().rstrip('\/').split('/')[-1].strip()
    return full_path.split(last_dir)[-1].strip()

# Setting up some constants
correction      = 0.2 # this is for image left and right correction
top_crop        = 70 # cropping image from above
bot_crop        = 25 # cropping image from below

# Generator funciton to work on batch of samples
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augmented_images, augmented_angles = [], []
            base_path = FLAGS.data_path.strip().rstrip('\/') + '/'
            for batch_sample in batch_samples:
                half_path = get_last_half_path(batch_sample[0]) # centre img
                images.append(cv2.imread(base_path + half_path))
                half_path = get_last_half_path(batch_sample[1]) # left img
                images.append(cv2.imread(base_path + half_path))
                half_path = get_last_half_path(batch_sample[2]) # right img
                images.append(cv2.imread(base_path + half_path))

                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                angles.extend([center_angle, left_angle, right_angle])

            # augmenting input images and output angles for training
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)

def main(_):
    print("data_path=", FLAGS.data_path)

    train_samples, validation_samples = load_data_samples()
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # row, col, ch = 160 - top_crop - bot_crop, 320, 3  # Trimmed image format
    row, col, ch = 160, 320, 3  # Full image format: (y, x, depth/channel)

    # Setting up nVidia ConvNet for training
    model = Sequential()
    # normalizing training images
    model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(row,col,ch), output_shape=(row,col,ch)))
    # trim image to only see section with road
    model.add(Cropping2D(cropping=((top_crop,bot_crop),(0,0)), input_shape=(row,col,ch)))
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

    # MSE (Mean Squared Error) for loss function and Adam as optimizer is used
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                validation_data=validation_generator, \
                nb_val_samples=len(validation_samples), nb_epoch=7)

    model.save('model.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
