# This is the main function in this folder

# import packages
import os
import pandas as pd
import numpy as np 
import os


# the loop
while(1) :
    flag = input('Please input the function you want to do >>> ')
    if flag == '0':
        print('get this help page >>>')
        print(' 1: build up the workload_pro')
        print(' 2: build up the workload')
        print(' 3: train the model')
        print(' 4: get the performance')
    if flag == '1':
        print('build up the workload_pro')
        
        file_name = input('[workload_generator_cat.py] Please input the file_name you want to build workload >>> ')
        sample_size = input('[workload_generator_num.py] Please input the sample_size of num >>> ')
        query_size = input('[get_query_pro.py] Please input the query_size >>> ')
        min_bound = input('[get_query_pro.py] Please input the min_bound >>> ')
        max_bound = input('[get_query_pro.py] Please input the max_bound >>> ')

        # os.system('python workload_generator_cat.py ' + file_name)
        """ workload_generator_cat.py get all combination of possible categories and save them in a csv file
        input: file_name string 
        output: csv file 
        notes: input should be the name of a csv file """

        os.system('python workload_generator_num.py ' + file_name + ' ' + sample_size)
        """ workload_generator_num.py get combination of possible numeric data of sample_size and save them in two csv file 
        input: file_name string, sample_size string
        output: sample_center.csv, sample_width.csv """

        os.system('python get_query_pro.py cat_comb_' + file_name + ' sample_center_' + file_name + ' sample_width_' + file_name + ' ' + query_size + ' ' + file_name + ' ' + min_bound + ' ' + max_bound)
        """ get_query.py get workload
        python get_query.py cat_comb_census13.csv sample_center_census13.csv sample_width_census13.csv 1000000000 census13.csv
        input: file_name string, query_size string
        output: query1000000000_filted_census13.npy, query1000000000_zero_census13.npy  """

    if flag == '2':
        print('build up the workload')
        
        file_name = input('[workload_generator_cat.py] Please input the file_name you want to build workload >>> ')
        query_size = input('[get_query.py] Please input the query_size >>> ')
        min_bound = input('[get_query.py] Please input the min_bound >>> ')
        max_bound = input('[get_query.py] Please input the max_bound >>> ')
        os.system('python get_query.py sample_width_' + file_name + ' ' + file_name + ' ' + query_size + ' ' + min_bound + ' ' + max_bound)

    if flag == '3':
        print('train the model')


    if flag == '4':
        print('get the performance')

         