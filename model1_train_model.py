# import packages
import sys
import numpy as np

import tensorflow as tf 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# some configurations
# X_train_path = 'X_train_200000.npy'
# y_train_path = 'y_train_200000.npy'
# model_id = 26
# batch_size = 128
# epochs = 20

X_train_path = sys.argv[1]
y_train_path = sys.argv[2]
model_id = int(sys.argv[3])
batch_size = int(sys.argv[4])
epochs = int(sys.argv[5])

X_train = np.load(X_train_path,allow_pickle=True).astype('float64')
y_train = np.load(y_train_path,allow_pickle=True).astype('float64')

# some functions
def build_model1(input_shape):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256,input_dim=input_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(10,input_dim=256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1,input_dim=10))
    return model
input_shape = (130,13)
model1 = build_model1(input_shape)
model1.compile(loss='mse',optimizer=Adam())
model1.fit(
    x = X_train,
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1,
)
model1.save('model1_' + str(model_id) + '.h5')