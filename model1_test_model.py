# import packages
import sys
import time
import numpy as np
import pandas as pd 

import tensorflow as tf 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 
from keras.models import load_model

# some configurations
# model_file = 'model1_26.h5'
# X_test_file = 'X_test_4884.2.npy'
# y_test_file = 'y_test_4884.2.npy'
# X_test_file = 'X_train_43957.8.npy'
# y_test_file = 'y_train_43957.8.npy'

model_file = sys.argv[1]
X_test_file = sys.argv[2]
y_test_file = sys.argv[3]

model1 = load_model(model_file)
X_test = np.load(X_test_file,allow_pickle=True).astype('float64')
y_test = np.load(y_test_file,allow_pickle=True).astype('float64')
sample_size = len(y_test)

# some functions
def Cal_q_err_origin(pred, gt):
    if pred >= gt:
        return pred/gt
    return gt/pred
def Cal_q_err(pred, gt):
    if min(pred, gt) == 0:
        if max(pred, gt) == 0:
            return 1
        else:
            return max(pred,gt)
    else:
        return Cal_q_err_origin(pred, gt)
def get_q_err_list(ture_label,fake_label):
    if len(ture_label) != len(fake_label) :
        print('get_reciprocal_q_error_list error: len')
        return
    else:
        q_err_list = []
        for i in range(len(ture_label)):
            q_err_list.append(Cal_q_err(fake_label[i],ture_label[i]))
    return q_err_list
""" get_y_pred
input: 
    y_pred: np.array the output of model
output:
    y_pred: np.array
note:
    output of model can not be used as prediction directly, sometimes output of model is less than 1.
    map the output of model to more tha  1 """
def get_y_pred(y_pred):
    filt = y_pred < 0
    y_pred[filt] = 0
    y_pred = [int(i) for i in y_pred]
    return y_pred

time1 = time.time()
try: 
    y_pred = model1.predict(X_test)
except:
    y_pred = model1(X_test)
time2 = time.time()

print('mean_squated_error:',mean_squared_error(y_test,y_pred))
print('time:',(time2-time1)*1000,' mean_time:',(time2-time1)*1000/sample_size)

y_pred = np.reshape(y_pred,(sample_size))
y_pred = get_y_pred(y_pred)
y_test = np.reshape(y_test,(sample_size)).astype(int)
q_error = get_q_err_list(y_test,y_pred)

print('test_size:',len(q_error))
result = pd.DataFrame({'y_test':y_test,'y_pred':y_pred,'q_error':q_error})
result.to_csv('result.csv',index=0)