from functools import reduce
import sys
import pandas as pd 

file_name = sys.argv[1]
def two_list_combination(list1, list2):
    return [str(i) + ',' + str(j) for i in list1 for j in list2]
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
cat,num,label_list = get_label_list(file_name)

cat_comb = reduce(two_list_combination, label_list)
cat_comb = [i.split(',') for i in cat_comb]
cat_comb = pd.DataFrame(cat_comb)
cat_comb.columns = cat
cat_comb.to_csv('cat_comb_'+file_name,index=0)