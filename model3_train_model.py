# import packages
import sys
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import time

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 

import tensorflow as tf 
import keras
from keras.models import load_model, Model, Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import pickle

# some configurations
model1_file = 'model1_26.h5'
file_path = 'census13.csv'
model_id = 26
batch_size = 128
epochs = 20000
X_train_z_path = 'model3_X_train_z_180000.0.npy'
X_train_q_path = 'model3_X_train_q_180000.0.npy'
y_train_path = 'model3_y_train_180000.0.npy'


# model1_file = sys.argv[1]
# file_path = sys.argv[2]
# model_id = int(sys.argv[3])
# batch_size = int(sys.argv[4])
# epochs = int(sys.argv[5])
# X_train_z_path = sys.argv[6]
# X_train_q_path = sys.argv[7]
# y_train_path = sys.argv[8]


def load_data(file_path):
    label_dic = {}
    data = pd.read_csv(file_path)
    columns = data.columns
    for column in columns:
        if data[column].dtype == 'object':
            le = preprocessing.LabelEncoder()
            le.fit(data[column])
            data[column] = le.transform(data[column])
            label_dic[column] = le
    data = np.array(data)
    return data,label_dic,columns
np_data,label_dic,columns = load_data(file_path)

input_dim = 100
input_shape = (2,len(columns))
data_shape = (batch_size,len(columns)) 
flatten_data_shape = batch_size * len(columns)
rate = len(np_data) / batch_size

print('input_dim:',input_dim)
print('input_shape:',input_shape)
print('data_shape:',data_shape)
print('data_shape[0]*data_shape[1]:',data_shape[0]*data_shape[1])

model1 = load_model(model1_file)
X_train_z = np.load(X_train_z_path)
X_train_q = np.load(X_train_q_path)
y_train = np.load(y_train_path)

# some functions
def build_generator(input_dim): 
    # input of model3 should be input into generator but generator only take z as its input
    # so the first layer of the generator filt the query out of model 
    model = Sequential()
    model.add(Dense(512,input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256,input_dim=512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256,input_dim=256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256,input_dim=256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256,input_dim=256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(flatten_data_shape,input_dim=256)) 
    model.add(Reshape(data_shape,input_dim=flatten_data_shape))
    return model
generator = build_generator(input_dim)
def build_model3(input_dim, input_shape, model1, generator, rate):
    z_input = Input(input_dim,dtype='int32',name='z_input')
    query_input = Input(input_shape,dtype='float32',name='query_input')
    generator = build_generator(input_dim)
    
    fake_data = generator(z_input)

    data_workload = tf.concat([fake_data, query_input], axis = 1)
    
    print('data_workload.shape:',data_workload.shape)
    
    model1._name = 'client1'
    model1.trainble = False
    ce = model1(data_workload) * rate
    model3 = Model(inputs=[z_input,query_input], outputs=ce)
    return model3

model3 = build_model3(input_dim, input_shape, model1, generator, rate)
model3.compile(loss='mse',optimizer=Adam())
history = model3.fit(
    x = [X_train_z, X_train_q],
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1,
)
with open('trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

generator.save('generator_' + str(model_id) + '.h5')
model3.save('model3_' + str(model_id) + '.h5')
