"""
Simple script to demonstrate how the data set can be loaded and a model can be trained.
You can, but you don't have to use this example.
Adapt this script as you want to build a more complex model, do pre processing, design features, ...

Author: Tano Müller

"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

import random
import skimage as sk
from skimage import transform
from skimage import util

def create_cnn_regress(width=128, height=128, depth=3, filters=(16, 32, 64)):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
 
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
 
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (5, 5), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(64)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
 
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(16)(x)
	x = Activation("relu")(x)
 
	# regression node should be added
	x = Dense(4, activation="linear")(x)
 
	# construct the CNN
	model = Model(inputs, x)
	opt = Adam(lr=1e-3, decay=1e-3 / 200)
	model.compile(loss="mean_absolute_error", optimizer=opt)
 
	# return the CNN
	return model	

def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def main():
	"""
    code snippet to load the data and train a model
    :return:
    """
	data_directory = ""
	x_train = np.load(os.path.join(data_directory, "x_train.npy"))
	y_train = np.load(os.path.join(data_directory, "y_train.npy"))

    ####################################################################################################################
    # do some pre processing
    ####################################################################################################################
	print("shapes from the raw train data:", x_train.shape, y_train.shape)

	# the number of new images to generate
	num_new_images = 1000


	# dictionary of the transformations functions we defined earlier
	available_transformations = {
		'rotate': random_rotation,
		'noise': random_noise,
		'horizontal_flip': horizontal_flip
	}

	for i in range(num_new_images):
		
		image_to_transform_idx = random.randint(0, num_new_images)
		image_to_transform = x_train[image_to_transform_idx][:][:][:]
		image_to_transform_label = y_train[image_to_transform_idx][:]

		# random num of transformations to apply
		num_transformations_to_apply = random.randint(1, len(available_transformations))
		num_transformations = 0
		transformed_image = None
		while num_transformations <= num_transformations_to_apply:
			# choose a random transformation to apply for a single image
			key = random.choice(list(available_transformations))
			transformed_image = available_transformations[key](image_to_transform)
			num_transformations += 1

		transformed_image = transformed_image[None, ...]
		image_to_transform_label = image_to_transform_label[None, ...]

		x_train = np.concatenate((x_train, transformed_image))
		y_train = np.concatenate((y_train, image_to_transform_label))

	print("shapes from the train data after transformation:", x_train.shape, y_train.shape)

    ####################################################################################################################
    # build your model
    ####################################################################################################################
	model = KerasRegressor(build_fn = create_cnn_regress, nb_epoch=200, batch_size=16, verbose=False)
 
	# train the model
	print("[INFO] training model...")
	model.fit(x_train, y_train)

    # save the model
	pickle.dump(model, open("model02.pkl", "wb"))
	print("created and saved the model")


if __name__ == '__main__':
    main()
