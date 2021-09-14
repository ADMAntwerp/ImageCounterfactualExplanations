# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:47:51 2020

@author: TVermeire
"""

import pandas as pd
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_target_experiment\output')



#%% Classes to consider
    
path = r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_target_experiment\output'
classes = listdir(path)


#%% TOTAL TABLE

table_total_mean = pd.DataFrame(columns=['Image', 'k1', 's1', 'ct1', 'tc1', 'k2', 's2', 'ct2', 'tc2'])
table_total_random = pd.DataFrame(columns=['Image', 'k1', 's1', 'ct1', 'tc1', 'k2', 's2', 'ct2', 'tc2'])
table_total_inpaint = pd.DataFrame(columns=['Image', 'k1', 's1', 'ct1', 'tc1', 'k2', 's2', 'ct2', 'tc2'])
table_total_blur = pd.DataFrame(columns=['Image', 'k1', 's1', 'ct1', 'tc1', 'k2', 's2', 'ct2', 'tc2'])


for class_name in classes:
    
    table_mean = pd.read_excel(class_name + '/' + class_name + '_mean.xlsx', index_col=0)
    table_random = pd.read_excel(class_name + '/' + class_name + '_random.xlsx', index_col=0)
    table_inpaint = pd.read_excel(class_name + '/' + class_name + '_inpaint.xlsx', index_col=0)
    table_blur = pd.read_excel(class_name + '/' + class_name + '_blur.xlsx', index_col=0)

    
    table_total_mean = table_total_mean.append(table_mean, ignore_index=True)
    table_total_random = table_total_random.append(table_random, ignore_index=True)
    table_total_inpaint = table_total_inpaint.append(table_inpaint, ignore_index=True)
    table_total_blur = table_total_blur.append(table_blur, ignore_index=True)
    
#%%
table = table_total_mean
#%% Effectiveness metrics

print('k1 MED: ' + str(np.nanmedian(table['k1'])))
print('k1 mu: ' + str(np.nanmean(table['k1'])))
print('k2 MED: ' + str(np.nanmedian(table['k2'])))
print('k2 mu: ' + str(np.nanmean(table['k2'])))

print('ct1 MED: ' + str(np.nanmedian(table['ct1'])))
print('ct1 mu: ' + str(np.nanmean(table['ct1'])))
print('ct2 MED: ' + str(np.nanmedian(table['ct2'])))
print('ct2 mu: ' + str(np.nanmean(table['ct2'])))

print('coverage 1: ' + str(1 - len(np.argwhere(np.isnan(table['k1'])))/len(table)))
print('coverage 2: ' + str(1 - len(np.argwhere(np.isnan(table['k2'])))/len(table)))



#%% Compare explanations

modes = ['mean', 'random', 'inpaint', 'blur']
columns1 = ['s1_' + mode for mode in modes]
columns2 = ['s2_' + mode for mode in modes]
segments_table = pd.DataFrame(columns = columns1 + columns2)

segments_table['s1_mean'] = table_total_mean['s1']
segments_table['s2_mean'] = table_total_mean['s2']
segments_table['s1_random'] = table_total_random['s1']
segments_table['s2_random'] = table_total_random['s2']
segments_table['s1_inpaint'] = table_total_inpaint['s1']
segments_table['s2_inpaint'] = table_total_inpaint['s2']
segments_table['s1_blur'] = table_total_blur['s1']
segments_table['s2_blur'] = table_total_blur['s2']

#%%


#%% Similarity of explanations

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

from ast import literal_eval

#%%
compared_modes = ['inpaint', 'blur']
jd_total = 0
number = 0
for i in range(len(table)):
    first_element = segments_table['s2_' + compared_modes[0]][i]
    second_element = segments_table['s2_' + compared_modes[1]][i]
    if isinstance(first_element, str) and isinstance(second_element, str):  
        first_element = str(first_element).replace('[ ', '[')
        second_element = str(second_element).replace('[ ', '[')
        first_element = str(first_element).replace('  ', ' ')
        second_element = str(second_element).replace('  ', ' ')
        jd = jaccard(literal_eval(str(first_element).replace(' ', ',')), literal_eval(str(second_element).replace(' ', ',')))
        jd_total += jd
        number += 1
        jd_avg = jd_total/number
