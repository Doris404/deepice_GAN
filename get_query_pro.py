import sys
import psycopg2
import random
import numpy as np
import pandas as pd 

cat_file = sys.argv[1]
center_file = sys.argv[2]
width_file = sys.argv[3]
query_size = int(sys.argv[4])
file_path = sys.argv[5]
min_bound = int(sys.argv[6])
max_bound = int(sys.argv[7])

cat = pd.read_csv(cat_file)
center = pd.read_csv(center_file)
width = pd.read_csv(width_file)
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

# print('cat_list:',cat_list)
# print('num_list:',num_list)
# print('label_list:',label_list)

def get_query(cat_tmp, center_tmp, width_tmp):
    query = ""
    for i in range(len(cat_list)):
        query += cat_list[i]
        query += "='"
        query += cat_tmp[i] + "'"
        query += " and "
    for i in range(len(center.columns)):
        query += center.columns[i]
        query += '>'
        query += str(center_tmp[i]-width_tmp[i])
        query += ' and '
        query += center.columns[i]
        query += '<'
        query += str(center_tmp[i]+width_tmp[i])
        query += ' and '
    return query[:-5]

def generate_workload(cat, center, width, query_size):
    cat_np = np.array(cat)
    center_np = np.array(center)
    width_np = np.array(width)
    
    cat_length = len(cat_np) - 1
    center_length = len(center_np) - 1
    width_length = len(width_np) - 1
    
    workload = np.array([])
    for i in range(query_size):
        tmp = random.randint(0,cat_length)
        cat_tmp = cat_np[tmp]
        
        tmp = random.randint(0,center_length)
        center_tmp = center_np[tmp]
        
        tmp = random.randint(0,width_length)
        width_tmp = width_np[tmp]
        
        query = get_query(cat_tmp,center_tmp,width_tmp)
        workload = np.append(workload,query)
    return workload

workload = generate_workload(cat,center,width,query_size)
print('len(workload):',len(workload))


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

gt = get_query_gt(file_path,workload)
# input data: pandas.series, min_bound: float, max_bound: float
# output data: np.array
def filtted_data(data,min_bound,max_bound):
    data = np.array(data)
    filt1 = data >= min_bound
    filt2 = data < max_bound
    filt = filt1 & filt2
    return filt
filt = np.array(filtted_data(pd.Series(gt),min_bound,max_bound))


pd.DataFrame({'query':workload[filt],'ground truth':gt[filt]}).to_csv(str(query_size) + "_" + str(min_bound)+ "_" + str(max_bound)+ "_"  + file_path)