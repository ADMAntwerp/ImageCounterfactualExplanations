# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:52:27 2020

@author: TVermeire
"""

import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt



#%%

path = 'C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/misclassifications_experiment/blur replacement 2/'
folderList = listdir(path)

#%%

data = pd.DataFrame()
for name in folderList: 
    data = data.append(pd.read_excel(path + '/' + name + '/' + name + '.xlsx', index_col = 0, sheet_name='Sheet1'))

#%%
    
np.sum(data['# images'])
np.sum(data['# misclassifications'])
#%%

data_misclassified = pd.DataFrame()
for name in folderList: 
    data_misclassified = data_misclassified.append(pd.read_excel(path + '/' + name + '/' + name + '.xlsx', index_col = 0, sheet_name='Sheet2'))
    
#%%
np.sum(data_misclassified['# misclassified'])
#%%

data_too_long = pd.DataFrame()
for name in folderList:
    data_too_long = data_too_long.append(pd.read_excel(path + '/' + name + '/' + name + '.xlsx', index_col = 0, sheet_name='Sheet3'))


#%%
    
np.count_nonzero(data_too_long['target score change'] < data_too_long['original class score change'])
np.argwhere(data_too_long['target score change'] < data_too_long['original class score change'])
np.count_nonzero(data_too_long['target score change'] > 0)
np.count_nonzero(data_too_long['original class score change'] > 0)


np.corrcoef(data_too_long['target score change'], data_too_long['original class score change'])

np.count_nonzero(data_too_long['original class'] == data_too_long['current class'])
