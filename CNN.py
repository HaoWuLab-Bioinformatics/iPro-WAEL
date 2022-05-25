import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
import os
import sys
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Bidirectional
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Reshape
from keras.constraints import maxnorm
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
import os
import random
from keras.layers import LSTM

def set_seed(seed):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def cnn(x_train, y_train, x_val, y_val, x_test, x_weight, learning_rate, INPUT_SHAPE, KERNEL_NUM, KERNEL_SIZE,
        LOSS, MAX_EPOCH, BATCH_SIZE):
    #set_seed(10)
    model = Sequential()
    model.add(Conv1D(KERNEL_NUM, kernel_size=KERNEL_SIZE, strides=1, activation='relu', padding='same', input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D())

    model.add(Conv1D(KERNEL_NUM, kernel_size=KERNEL_SIZE, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D())

    model.add(Conv1D(KERNEL_NUM, kernel_size=KERNEL_SIZE, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    filepath = "model_cnn.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
    model.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                       validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping_monitor])
    model_cnn = load_model(filepath)
    weight_proba = model_cnn.predict(x_weight)
    test_proba = model_cnn.predict(x_test)
    test_class = model_cnn.predict_classes(x_test)
    return weight_proba, test_proba, test_class


def pred(x_train, y_train, x_weight, x_test, learning_rate, KERNEL_NUM, KERNEL_SIZE):
    x_train = np.expand_dims(x_train, 2)
    x_weight = np.expand_dims(x_weight, 2)
    x_test = np.expand_dims(x_test, 2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1 / 7, random_state=10)
    print(x_train.shape)
    print(x_val.shape)
    print(x_weight.shape)
    print(x_test.shape)

    LOSS = 'binary_crossentropy'
    MAX_EPOCH = 100
    BATCH_SIZE = 50
    INPUT_SHAPE = x_train.shape[1:3]
    print(INPUT_SHAPE)
    weight_proba, test_proba, test_class = cnn(x_train, y_train, x_val, y_val, x_test, x_weight, learning_rate,
                                               INPUT_SHAPE, KERNEL_NUM, KERNEL_SIZE, LOSS, MAX_EPOCH, BATCH_SIZE)
    return weight_proba, test_proba, test_class
