# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:22:32 2020

@author: TVermeire
"""

import pandas as pd
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_experiment\output')


#%% Classes to consider
    
path = r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_experiment\output'
classes = listdir(path)

#%% INDIVIDUAL SUMMARIES MEAN

for class_name in classes:
    
    table = pd.read_excel(class_name + '/' + class_name + '.xlsx')
    
    statistics = pd.Series(['k', 'ct', '% found'])
    modes = ['mean', 'blur', 'random', 'inpaint']
    
    summary_table = dfObj = pd.DataFrame(columns=['statistic', 'mean', 'blur', 'random', 'inpaint'], index = [i for i in range(len(statistics))])
    
    summary_table['statistic'] = statistics
    
    for mode in modes: 
        summary_table[mode][0] = np.nanmean(table['k_' + mode])
        summary_table[mode][1] = np.nanmean(table['ct_' + mode])
        summary_table[mode][2] = 1 - len(table[np.isnan(table['k_' + mode])]) / len(table)     
    # Output table    
    os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_experiment/output/'+ class_name)
    with pd.ExcelWriter(class_name + ' - summary - mean.xlsx') as writer: 
            summary_table.to_excel(writer)
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_experiment\output')

#%% INDIVIDUAL SUMMARIES MEDIAN

for class_name in classes:
    
    table = pd.read_excel(class_name + '/' + class_name + '.xlsx')
    
    statistics = pd.Series(['k', 'ct', '% found'])
    modes = ['mean', 'blur', 'random', 'inpaint']
    
    summary_table = dfObj = pd.DataFrame(columns=['statistic', 'mean', 'blur', 'random', 'inpaint'], index = [i for i in range(len(statistics))])
    
    summary_table['statistic'] = statistics
    
    for mode in modes: 
        summary_table[mode][0] = np.nanmedian(table['k_' + mode])
        summary_table[mode][1] = np.nanmedian(table['ct_' + mode])
        summary_table[mode][2] = 1 - len(table[np.isnan(table['k_' + mode])]) / len(table)
        
    # Output table    
    os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_experiment/output/'+ class_name)
    with pd.ExcelWriter(class_name + ' - summary - median.xlsx') as writer: 
            summary_table.to_excel(writer)
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_experiment\output')



    
#%% TOTAL TABLE

table_total = pd.DataFrame(columns=['Image', 'k_mean', 's_mean', 'ct_mean', 'nc_mean', 'k_blur', 's_blur', 'ct_blur', 'nc_blur', 'k_random', 's_random', 'ct_random', 'nc_random', 'k_inpaint', 's_inpaint', 'ct_inpaint', 'nc_inpaint'])


for class_name in classes:
    
    table_individual = pd.read_excel(class_name + '/' + class_name + '.xlsx', index_col=0)
    
    table_total = table_total.append(table_individual, ignore_index=True)
   
#%% TOTAL TABLE SUMMARY MEAN
statistics = pd.Series(['k', 'ct', '% found'])  
modes = ['mean', 'blur', 'random', 'inpaint'] 
summary_table = dfObj = pd.DataFrame(columns=['statistic', 'mean', 'blur', 'random', 'inpaint'], index = [i for i in range(len(statistics))])
 
summary_table['statistic'] = statistics
    
for mode in modes: 
    summary_table[mode][0] = np.nanmean(table_total['k_' + mode])
    summary_table[mode][1] = np.nanmean(table_total['ct_' + mode])
    summary_table[mode][2] = 1 - len(table_total[np.isnan(table_total['k_' + mode])]) / len(table_total)
    
# Output table    
os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_experiment\output')
with pd.ExcelWriter('total - summary - mean.xlsx') as writer: 
        summary_table.to_excel(writer)
#%% TOTAL TABLE SUMMARY MEDIAN
statistics = pd.Series(['k', 'ct', '% found'])  
modes = ['mean', 'blur', 'random', 'inpaint'] 
summary_table = dfObj = pd.DataFrame(columns=['statistic', 'mean', 'blur', 'random', 'inpaint'], index = [i for i in range(len(statistics))])
 
summary_table['statistic'] = statistics
    
for mode in modes: 
    summary_table[mode][0] = np.nanmedian(table_total['k_' + mode])
    summary_table[mode][1] = np.nanmedian(table_total['ct_' + mode])
    summary_table[mode][2] = 1 - len(table_total[np.isnan(table_total['k_' + mode])]) / len(table_total)
    
# Output table    
os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_experiment\output')
with pd.ExcelWriter('total - summary - median.xlsx') as writer: 
        summary_table.to_excel(writer)
    
        
#%% Only images with 4 explanations found
        
table_new_class_3 = table_total[['nc_mean', 'nc_inpaint', 'nc_blur']]
table_new_class_3.dropna(inplace=True)
table_new_class_3.reset_index(inplace=True)

#%% ADDITIONAL ANALYSIS: number of new classes
# only include images for which 4 explanation are found
modes = ['mean', 'inpaint', 'blur']
number_of_classes = pd.Series(index = [i for i in range(len(table_new_class_3))])
for i in range(len(table_new_class_3)):
    new_classes = []
    for mode in modes:
        new_class = table_new_class_3['nc_' + mode][i]
        new_classes.append(new_class)
    number_of_classes[i] = len(np.unique(new_classes))
        
number_of_classes.value_counts(normalize=True, sort=False)


#%% Correlation k and ct

table_total_4[['k_mean', 'k_blur', 'k_random', 'k_inpaint']].corr()
table_total_4[['ct_mean', 'ct_blur', 'ct_random', 'ct_inpaint']].corr()


#%% Similarity of explanations

def calculate_similarity(segments_explanations):
    union = np.unique(segments_explanations)
    intersection = []
    for i in union:
        counter = 0
        for j in segments_explanations:
            if i in j: 
                counter += 1
        if counter == len(segments_explanations):
            intersection.append(i)
    similarity = len(intersection)/len(union)
    return similarity

explanations_similarity = pd.Series(index = [i for i in range(len(table_total_4))])
for i in range(len(table_total_4)):
    explanations = []
    for mode in modes: 
        explanations.append(table_total_4['s_' + mode][i])
    explanations_similarity[i] = calculate_similarity(explanations)
        

