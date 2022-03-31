""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import LaneNet


# Load training images
# 加载训练文件图像
train_images = pickle.load(open("archive/full_CNN_train.p", "rb"))

# Load image labels
# 加载图像标签
labels = pickle.load(open("archive/full_CNN_labels.p", "rb"))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size = 0.1)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 128
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]
lane = LaneNet(input_shape, pool_size)
lane.createmodel()
lane.buildModel(X_train, y_train, X_val, y_val, save = "Model/LaneNet.h5", summary = True)
