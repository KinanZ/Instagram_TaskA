"""
Simple script to demonstrate how the data set can be loaded and a prediction can be made.
You can, but you don't have to use this example.
Adapt this script as you want to build a more complex model, do pre processing, design features, ...

Author: Tano Müller

"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Train import create_cnn_regress
import pickle


def main():
    """
    code snippet to load and evaluate the model
    :return:
    """
    data_directory = ""
    x_valid = np.load(os.path.join(data_directory, "x_valid.npy"))
    y_valid = np.load(os.path.join(data_directory, "y_valid.npy"))

    ####################################################################################################################
    # do some pre processing
    ####################################################################################################################

    ####################################################################################################################
    # make predictions
    ####################################################################################################################
    # load the model
    model = pickle.load(open("model02.pkl", "rb"))
    # create the predictions
    prediction = model.predict(x_valid)
    # calculate several metrics
    mae = mean_absolute_error(y_valid, prediction)
    mse = mean_squared_error(y_valid, prediction)

    # get some info
    print("several performance metrics:")
    print("mae:", mae)
    print("mse:", mse)


if __name__ == '__main__':
    main()
