# import packages
import pandas as pd 
import numpy as np
import sys
import os 
import time
import psycopg2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

file_path = 'census13.csv'
z_dim = 100
data_size = 200000
query_path = 'filt_10000_0_1000000_census13.csv'
train_test_rate = 0.1

# file_path = sys.argv[1]
# z_dim = int(sys.argv[2])
# data_size = int(sys.argv[3])
# query_path = sys.argv[4]
# train_test_rate = float(sys.argv[5])

workload = pd.read_csv(query_path)['query']
np_workload = np.array(workload)
query_size = len(np_workload)

# some functions
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
        cardinality = answer[0][0]
    conn.close()
    return cardinality

X_train_z = []
X_train_query = []
y_train = np.array([])

for i in range(data_size):
    z = np.random.random(z_dim)
    idx = np.random.randint(0,query_size,1)
    query0 = np_workload[idx]
    query = transformer_query(query0,label_dic)
    cardinality = get_cardinality(np_data,query0,columns)
    X_train_z.append(z)
    X_train_query.append(query)
    y_train = np.append(y_train, cardinality)
    # break

X_train_z = np.array(X_train_z)
X_train_query = np.array(X_train_query)

X_train_z, X_test_z, y_train, y_test = train_test_split(X_train_z, y_train, test_size = train_test_rate, random_state=42)
X_train_query, X_test_query = train_test_split(X_train_query, test_size = train_test_rate, random_state=42)

np.save('model3_X_train_z_' + str(data_size * (1 - train_test_rate)) + '.npy',X_train_z)
np.save('model3_X_train_q_' + str(data_size * (1 - train_test_rate)) + '.npy',X_train_query)
np.save('model3_y_train_' + str(data_size * (1 - train_test_rate)) + '.npy',y_train)

np.save('model3_X_test_z_' + str(data_size * train_test_rate) + '.npy',X_test_z)
np.save('model3_X_test_q_' + str(data_size * train_test_rate) + '.npy',X_test_query)
np.save('model3_y_test_' + str(data_size * train_test_rate) + '.npy',y_test)

print(X_train_z.shape)
print(X_train_query.shape)
print(y_train.shape)
print(X_test_z.shape)
print(X_test_query.shape)
print(y_test.shape)