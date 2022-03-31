import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class LaneNet:
    def __init__(self, input_shape, pool_size):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.model = keras.models.Sequential()

    def createmodel(self):

        self.model.add(keras.layers.BatchNormalization(input_shape = self.input_shape))
        self.model.add(keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv1"))
        self.model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv2"))
        self.model.add(keras.layers.MaxPool2D(pool_size = self.pool_size))

        self.model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv3"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv4"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv5"))
        self.model.add(keras.layers.MaxPool2D(pool_size = self.pool_size))

        self.model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv6"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                           activation = "relu", name = "Conv7"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.MaxPool2D(pool_size = self.pool_size))

        self.model.add(keras.layers.UpSampling2D(size = self.pool_size))
        self.model.add(
                keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                             activation = "relu", name = "DeConv1"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(
                keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid",
                                             activation = "relu", name = "DeConv2"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.UpSampling2D(size = self.pool_size))

        self.model.add(
                keras.layers.Conv2DTranspose(32, (3, 3), padding = 'valid', strides = (1, 1), activation = 'relu',
                                             name = 'DeConv3'))
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(
                keras.layers.Conv2DTranspose(32, (3, 3), padding = 'valid', strides = (1, 1), activation = 'relu',
                                             name = 'DeConv4'))
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(
                keras.layers.Conv2DTranspose(16, (3, 3), padding = 'valid', strides = (1, 1), activation = 'relu',
                                             name = 'DeConv5'))
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.UpSampling2D(size = self.pool_size))

        self.model.add(
                keras.layers.Conv2DTranspose(16, (3, 3), padding = 'valid', strides = (1, 1), activation = 'relu',
                                             name = 'DeConv6'))

        self.model.add(keras.layers.Conv2DTranspose(1, (3, 3), padding = 'valid', strides = (1, 1), activation = 'relu',
                                                    name = 'Final'))

    def buildModel(self, x_train, y_train, x_val, y_val, batch_size = 128, epochs = 10, save = "", summary = False):
        # Using a generator to help the self.model use less data
        # Channel shifts help with shadows slightly
        datagen = keras.preprocessing.image.ImageDataGenerator(channel_shift_range = 0.2)
        datagen.fit(x_train)
        # Compiling and training the self.model
        self.model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
        # 使用 Python 生成器逐批生成的数据，按批次训练模型。
        self.model.fit_generator(
                datagen.flow(x_train, y_train, batch_size = batch_size),
                steps_per_epoch = len(x_train) / batch_size,
                epochs = epochs, verbose = 1, validation_data = (x_val, y_val))
        # Freeze layers since training is done
        if len(save) != 0:
            # 编译模型之前冻结所有权重
            self.model.trainable = False
            self.model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
            # Save self.model architecture and weights
            self.model.save(save)
        if summary:
            self.model.summary()


if __name__ == '__main__':
    """
    test code
    """
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
    lane.buildModel(X_train, y_train, X_val, y_val)
