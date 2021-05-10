import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model

from sklearn import preprocessing

generator_model = sys.argv[1]
sample_size = int(sys.argv[2])
real_data = sys.argv[3]
z_dim = 100

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

_,label_dic,columns = load_data(real_data)

def get_sample_data(generator,label_dic,columns,sample_size):
    
    # Sample random noise
    z = np.random.normal(0,1,(sample_size,z_dim))
    
    # Generate data from random noise
    gen_data = generator.predict(z)
    gen_data = gen_data.reshape(sample_size,-1)
    gen_data = pd.DataFrame(gen_data,columns=columns)

    gen_data.to_csv('raw/'+str(sample_size)+'.csv')
    return gen_data

generator = load_model(generator_model)
get_sample_data(generator,label_dic,columns,sample_size)
