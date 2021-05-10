# import packages
import os
import sys
import time
import numpy as np
import pandas as pd 
import psycopg2

import tensorflow as tf 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from keras.models import load_model

# some configuration
generator_file = 'generator_27.h5'
X_test_z_file = 'model3_X_test_20000.0_z.npy' # need change the name 
query_file = 'filt_10000_0_1000000_census13.csv'
sample_query_size = 20000
file_path = 'census13.csv'
sample_data_num = 100

# generator_file = sys.argv[1] 
# X_test_z_file = sys.argv[2] 
# query_file = sys.argv[3]
# sample_query_size = int(sys.argv[4])
# file_path = sys.argv[5]

generator = load_model(generator_file)
workload = pd.read_csv(query_file)['query']
np_workload = np.array(workload)
workload_size = len(np_workload)
y_test = np.array(pd.read_csv(query_file)['ground truth'])

# some function
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
data = pd.read_csv(file_path)
""" inverse_transformer
input: 
    gen_data: np.array data generate by generator
    label_dic: map of categorical data to its label
    columns: columns of original data
output:
    gen_data: np.array inverse column to its original state """
def inverse_transformer(gen_data, label_dic, columns):
    gen_data = pd.DataFrame(gen_data)
    gen_data.columns = columns
    for column in label_dic.keys():
        column_data = gen_data[column]
        scaler = MinMaxScaler.fit()
        le = label_dic[column]
        gen_data[column] = le.inverse_transform(gen_data[column])
    gen_data = np.array(gen_data)
    return gen_data
""" transformer_query
input:
    gen_query: np.array the original query
    label_dic:  the map of categorical data to its label
output:
    gen_query: np.array the transformed query """
def transformer_query(gen_query, label_dic):
    gen_query = gen_query[0].split(' and ')
    feature1 = np.array([])
    feature2 = np.array([])
    for i in range(len(gen_query)):
        item = gen_query[i]
        if '>' in item:
            item = gen_query[i].split('>')
            feature1 = np.append(feature1,float(item[1]))
        elif '<' in item:
            item = gen_query[i].split('<')
            feature2 = np.append(feature2,float(item[1]))
        elif ' = ' in item:
            item = gen_query[i].split(' = ')
            item[1] = label_dic[item[0]].transform([item[1][1:-1]])[0]
            feature1 = np.append(feature1,item[1])
            feature2 = np.append(feature2,item[1])
    return np.array([feature1,feature2])
""" get_cardinality
input:
    gen_data: np.array the original form of data
    gen_query: np.array the original form of query
ouput:
    cardinality: np.array of the same size as gen_query
note:
    run the gen_query on gen_data """
def get_cardinality(gen_data, gen_query, columns):
    cardinality = np.array([])

    gen_data = pd.DataFrame(gen_data)
    gen_data.columns = columns
    gen_data.to_csv('tmp.csv',index=0)
    
    conn = psycopg2.connect(database="lixiaotong", user="lixiaotong", password="123456", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    cur.execute("truncate table census13_fake;")
    cur.execute("copy census13_fake from '/home/xiaotong/xiaotong/Project/DeepICE/deepice_GAN_pro/tmp.csv' CSV HEADER;")
    for i in range(len(gen_query)):
        cur.execute("SELECT count(*) from census13_fake where " + gen_query[i] + ";" )
        answer = cur.fetchall()
        cardinality = np.append(cardinality,answer[0][0])
    conn.close()
    return cardinality
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

""" 
y_test = run query on fake data
y_pred = run query on true data """

idx = np.random.randint(0,workload_size,sample_query_size)
X_test_q0 = np_workload[idx]
y_test = y_test[idx]


""" generate fake data """
for i in range(sample_data_num):
    z = np.random.random((1,100))
    fake_data = np.array(generator(z))
    fake_data = pd.DataFrame(fake_data[0])
    fake_data.columns = columns
    fake_data.to_csv('raw/' + str(i) + '.csv')
    break

""" transformer """
rootdir = 'raw'
file_list = []
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件

for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path):
        file_list.append(path)
print('file_list:\n',file_list)

for path in file_list:
    iteration_data = pd.read_csv(path,index_col=0)
    gen_cat_data = {}
    for column in label_dic.keys():
        column_data = iteration_data[column]
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
        iteration_data[column] = gen_cat_data[column]
        iteration_data.to_csv('cat/'+path[4:],index=0)

for path in file_list:
    gen_data = np.array(pd.read_csv('cat/'+path[4:]))
    print(gen_data.shape)
    print(X_test_q0.shape)
    y_pred = get_cardinality(gen_data,X_test_q0,columns)
    print(len(y_pred))
    print(y_test.shape)
    q_error = get_q_err_list(y_test,y_pred)
    result = pd.DataFrame({'y_test':y_test,'y_pred':y_pred,'q_error':q_error})
    result.to_csv('model3_result.csv',index = 0)