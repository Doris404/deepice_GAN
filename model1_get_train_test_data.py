# import packages
import pandas as pd 
import numpy as np
import sys
import os 
import psycopg2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# some configurations
# file_path = 'census13.csv'
# batch_size = 128
# data_size = 200000
# query_path = 'filt_10000_0_1000000_census13.csv'
# train_test_rate = 0.1

file_path = sys.argv[1]
batch_size = int(sys.argv[2])
sample_size = int(sys.argv[3])
query_path = sys.argv[4]
train_test_rate = float(sys.argv[5])

data = pd.read_csv(file_path)
np_data = np.array(data)
data_size = len(np_data)
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
_,label_dic,columns = load_data(file_path)
""" transformer
input:
    gen_data: np.array the original data
    label_dic: the map of categorical data to its label
    columns: column names of original data
output:
    gen_data: np.array the tranformed data """
def transformer(gen_data, label_dic, columns):
    gen_data = pd.DataFrame(gen_data)
    gen_data.columns = columns
    for column in label_dic.keys():
        column_data = gen_data[column]
        le = label_dic[column]
        gen_data[column] = le.transform(gen_data[column])
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
        cardinality = answer[0][0]
    conn.close()
    return cardinality

X_train = []
y_train = np.array([])

for i in range(sample_size):
    idx = np.random.randint(0,data_size,batch_size)
    sample_data0 = np.array(np_data[idx])
    sample_data = transformer(sample_data0,label_dic,columns)
    
    idx = np.random.randint(0,query_size,1)
    sample_query0 = np.array(np_workload[idx])
    sample_query = transformer_query(sample_query0,label_dic)

    try:
        cardinality = get_cardinality(sample_data0, sample_query0, columns)
        data_workload = np.vstack((sample_data,sample_query))
        X_train.append(data_workload)
        y_train = np.append(y_train,cardinality)
    except:
        print('error:',i)
    # break
X_train = np.array(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = train_test_rate, random_state=42)

np.save('model1_X_train_' + str(batch_size) + '_' + str(data_size * (1 - train_test_rate)) + '.npy', X_train)
np.save('model1_y_train_' + str(batch_size) + '_' + str(data_size * (1 - train_test_rate)) + '.npy', y_train)
np.save('model1_X_test_' + str(batch_size) + '_' + str(data_size * train_test_rate) + '.npy', X_test)
np.save('model1_y_test_' + str(batch_size) + '_' + str(data_size * train_test_rate) + '.npy', y_test)