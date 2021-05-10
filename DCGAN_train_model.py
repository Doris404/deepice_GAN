# import some packages
import os
import csv
import sys
import pandas as pd
import numpy as np
import psycopg2

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 

import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam

# some configures
model_id = sys.argv[1] 

file_path = sys.argv[2]
z_dim = int(sys.argv[3])

workload_set = sys.argv[4]
sample_workload_size = int(sys.argv[5])
sample_size = int(sys.argv[6])

iterations = int(sys.argv[7])
batch_size = int(sys.argv[8])
sample_interval = iterations / 20

# file_path = 'census13.csv'
# z_dim = 100

# workload_set = 'query200000_census13.csv'
# sample_workload_size = 10
# sample_size = 1000

# iterations = 20000
# batch_size = 128
# sample_interval = 1000


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
data,label_dic,columns = load_data(file_path)

def transformer(gen_data):
    # input gen_data: np.array
    # output gen_data: np.array
    gen_data = pd.DataFrame(gen_data)
    gen_data.columns = columns
    gen_cat_data = {}
    for column in label_dic.keys():
        column_data = gen_data[column]
        le = label_dic[column]
        min_max_scaler = preprocessing.MinMaxScaler()
        column_data = np.array(column_data)
        column_data = column_data.reshape(-1,1)
        column_data = min_max_scaler.fit_transform(column_data)
        column_data = column_data * (len(label_dic[column].classes_) - 1)
        column_data = column_data.astype(int)
        column_data = le.inverse_transform(column_data)
        gen_cat_data[column] = column_data
    for column in label_dic.keys():
        gen_data[column] = gen_cat_data[column]
    gen_data = np.array(gen_data)
    return gen_data

def get_sample_workload(workload_set,sample_workload_size):
    # output: workload_sample[idx] np.array
    workload_set = pd.read_csv(workload_set)
    workload_sample = np.array(workload_set)
    length = len(workload_sample)
    idx = np.random.randint(0,length,sample_size)
    return workload_sample[idx]

def get_true_label(workload_sample):
    # output: true_label np.array
    workload_sample = pd.DataFrame(workload_sample)
    workload_sample.columns = ['index','query','ground truth']
    true_label = workload_sample['ground truth']
    true_label = np.array(true_label)
    return true_label

def get_fake_label(workload_sample, gen_data_tmp):
    gen_data_file = str(len(gen_data_tmp))+'.csv'
    gen_data_tmp = pd.DataFrame(gen_data_tmp)
    gen_data_tmp.columns = columns
    gen_data_tmp.to_csv(gen_data_file,index=0)

    conn = psycopg2.connect(database="lixiaotong", user="lixiaotong", password="123456", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    predict_label = []
    cur.execute("truncate table census13_fake;")
    os.system("copy census13_fake from '/home/xiaotong/xiaotong/Project/DeepICE/DCGAN3_model/"+gen_data_file+" CSV HEADER;")
    for i in range(len(workload_sample)):
        cur.execute("SELECT count (*) from census13_fake where " + workload_sample[i][1] + ";" )
        rows = cur.fetchall()
        predict_label.append(rows[0][0] * rate)
    cur.execute("truncate table census13_fake;")
    conn.close()
    return predict_label

def Cal_q_err(pred, gt):
    if min(pred, gt) == 0:
        if max(pred, gt) == 0:
            return 1
        else:
            return max(pred,gt)
    else:
        return Cal_q_err_origin(pred, gt)
def get_reciprocal_q_err_list(ture_label,fake_label):
    if len(ture_label) != len(fake_label) :
        print('get_reciprocal_q_error_list error: len')
        return
    else:
        reciprocal_q_err_list = []
        for i in range(len(ture_label)):
            reciprocal_q_err_list.append(1/Cal_q_err(fake_label[i],ture_label[i]))
    return np.expand_dims(reciprocal_q_err_list,1)

def get_fake(batch_size,workload_set,sample_workload_size,gen_data_tmp):
    workload_sample = get_sample_workload(workload_set,sample_workload_size)
    ture_label = get_true_label(workload_sample)
    fake_label = get_fake_label(workload_sample,gen_data_tmp)
    fake = get_reciprocal_q_err_list(ture_label,fake_label)
    return fake


data_rows = 1
data_cols = len(data[0])
channels = 1
data_cols = 13
data_shape = (data_rows, data_cols, channels)
rate = len(np.array(data) / sample_size)

def build_generator_DCGAN(z_dim):
    model = Sequential()
    
    model.add(Dense(256*7*7, input_dim=z_dim))
    model.add(Reshape((7,7,256)))
    
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    
    # Batch normalization
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Reshape((1,14*14*128)))
    model.add(Dense(1*13*1, input_dim=14*14*13))
    model.add(Reshape((1,13,1)))
    
    return model

def build_discriminator_DCGAN(data_shape):
    model = Sequential()
    
    model.add(Flatten())
    model.add(Dense(32*14*14, input_dim=1))
    model.add(Reshape((14,14,32)))
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=data_shape,padding='same'))
    
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=data_shape,padding='same'))
    
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)
    
    return model

# Build and compile the Discriminator
discriminator = build_discriminator_DCGAN(data_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# print('discriminator:',discriminator.summary())

# Build the Generator
generator = build_generator_DCGAN(z_dim)
# print('generator:',generator.summary())

# Keep Discriminator's parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discrimiantor to train the Generator
gan = build_gan(generator,discriminator)
gan.compile(loss='binary_crossentropy',optimizer=Adam())
# print('gan:',gan.summary())

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval, file_path):
    
    # Load data from file_path 
    data,_,_ = load_data(file_path)
    data = np.expand_dims(data,axis=1)
    data = np.expand_dims(data,axis=3)
    
    for iteration in range(iterations):
        
        # -----------------------
        # Train the Discriminator
        # -----------------------
        
        # Get a random batch of real data
        idx = np.random.randint(0,data.shape[0],batch_size)
        sample_data1 = data[idx]
        sample_data = sample_data1
        
        # Generate a batch of fake data
        z = np.random.normal(0,1,(batch_size,100))
        gen_data = generator.predict(z)
        gen_data_tmp = gen_data.reshape(batch_size,-1)
        gen_data_tmp = transformer(gen_data_tmp) 
        
        # Train Discriminator
        real = np.ones((batch_size,1))
        fake = get_fake(batch_size,workload_set,sample_workload_size,gen_data_tmp)
        fake = fake.astype('float64')

        d_loss_real = discriminator.train_on_batch(sample_data, real)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -------------------
        # Train the Generator
        # -------------------
        
        # Generate a batch of fake data
        z = np.random.normal(0,1,(batch_size,100))
        gen_data = generator.predict(z)
        
        # Train Generator
        g_loss = gan.train_on_batch(z, real)
        
        if (iteration+1) % sample_interval == 0:
            
            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss,g_loss))
            accuracies.append(100.0*accuracy)
            
train(iterations, batch_size, sample_interval,'census13.csv')

gan.save('DCGAN_' + file_path[:-4] + '_' + str(model_id)+'.h5')
discriminator.save('DCGAN_discriminator_' + file_path[:-4] + '_' + str(model_id)+'.h5')
generator.save('DCGAN_generator_' + file_path[:-4] + '_' + str(model_id)+'.h5')      
