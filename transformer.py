import os
import sys
import pandas as pd 
import numpy as np
from sklearn import preprocessing

real_data = sys.argv[1]

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
        iteration_data.to_csv('cat/'+path[4:])