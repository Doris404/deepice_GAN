import sys
import psycopg2
import random
import numpy as np
import pandas as pd 
from sklearn import preprocessing


width_file = sys.argv[1]
file_path = sys.argv[2]
query_size = int(sys.argv[3])
min_bound = int(sys.argv[4])
max_bound = int(sys.argv[5])

def get_label_list(file_name):
    cat = []
    num = []
    label_list = []
    data = pd.read_csv(file_name)
    columns = data.columns
    for column in columns:
        if (data[column].dtype == 'object'):
            cat.append(column)
        else:
            num.append(column) 
    for column in cat:
        label_list.append(data[column].unique())
    return cat, num, label_list
cat_list,num_list,label_list = get_label_list(file_path)
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
    return data,label_dic,columns

_,label_dic,columns = load_data(file_path)
data = pd.read_csv(file_path)
np_data = np.array(data)
width = pd.read_csv(width_file)
np_width = np.array(width)

len_data = len(np_data) - 1
len_width = len(np_width) - 1

# input: data_sample a line of data, width_tmp width range on numeric column
# output: a query
def get_query(data_sample, width_tmp):
    query = ''
    k = 0 
    for i in range(len(columns)):
        if columns[i] in num_list:
            """ do some operation """
            query += columns[i]
            query += '>'
            query += str(int(data_sample[i] - width_tmp[k]))
            query += ' and '
            query += columns[i]
            query += '<'
            query += str(int(data_sample[i] + width_tmp[k]))
            query += ' and '
            k += 1
        else:
            query += columns[i]
            query += " = '"
            query += str(data_sample[i]) + "'"
            query += " and "
    return query[:-5]

def get_query_exact(data_sample):
    query = ''
    k = 0 
    for i in range(len(columns)):
        if columns[i] in num_list:
            """ do some operation """
            query += columns[i]
            query += '='
            query += str(data_sample[i])
            query += ' and '
        else:
            query += columns[i]
            query += " = '"
            query += str(data_sample[i]) + "'"
            query += " and "
    return query[:-5]


# input: file_path original data, query_size the size of workload, width_file the file contians width
# output: workload np.array
def get_workload(file_path, query_size, width_file):
    workload = np.array([])

    for i in range(query_size):
        idx_data_sample = np.random.randint(0,len_data)
        idx_width_sample = np.random.randint(0,len_width)
        query = get_query(np_data[idx_data_sample], np_width[idx_width_sample])
        workload = np.append(workload,query)
        # query = get_query_exact(np_data[idx_data_sample])
        # workload = np.append(workload,query)
    return workload
# input: file_path string, workload string
# output: gt np.array
def get_query_gt(file_path,workload):
    gt = []
    conn = psycopg2.connect(database="lixiaotong", user="lixiaotong", password="123456", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    for query in workload:
        cur.execute("SELECT count (*) from " + file_path[:-4] + " where " + query + ";" )
        rows = cur.fetchall()
        gt.append(rows[0][0])
    conn.close()
    return np.array(gt)

# input data: pandas.series, min_bound: float, max_bound: float
# output data: np.array
def filtted_data(data,min_bound,max_bound):
    data = np.array(data)
    filt1 = data >= min_bound
    filt2 = data < max_bound
    filt = filt1 & filt2
    return filt

workload = get_workload(file_path, query_size, width_file)
gt = get_query_gt(file_path,workload)
filt = np.array(filtted_data(pd.Series(gt),min_bound,max_bound))

pd.DataFrame({'query':workload[filt],'ground truth':gt[filt]}).to_csv(str(query_size) + "_" + str(min_bound)+ "_" + str(max_bound)+ "_"  + file_path)