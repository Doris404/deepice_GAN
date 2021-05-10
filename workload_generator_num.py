from functools import reduce
import sys
import random
import numpy as np
import pandas as pd 

def get_mu_std(data,num):
    mu = {}
    std = {}
    for column in num:
        mu[column] = data[column].mean()
        std[column] = data[column].std()
    return mu,std

file_name = sys.argv[1]
sample_size = int(sys.argv[2])

data = pd.read_csv(file_name)
columns = data.columns
num = []

for column in columns:
    if (data[column].dtype != 'object'):
        num.append(column)
data_num = data[num]
data_num_np = np.array(data_num)
length = len(data_num_np) - 1

# generate the center
sample_center = []
for i in range(sample_size): 
    tmp = random.randint(0,length)
    sample_center.append(data_num_np[tmp])

width_list = []

# generate the width
mu,std = get_mu_std(data,num)
# Parallelization to be continued
for i in range(sample_size): 
    tmp = []
    for column in num:
        tmp.append(abs(random.normalvariate(mu[column],std[column])))
    width_list.append(tmp)

# save in csv
sample_center = pd.DataFrame(sample_center)
sample_center.columns = num
width_list = pd.DataFrame(width_list)
width_list.columns = num
sample_center.to_csv('sample_center_'+file_name,index=0)
width_list.to_csv('sample_width_'+file_name,index=0)