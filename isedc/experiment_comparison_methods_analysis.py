# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:53:32 2020

@author: TVermeire
"""

import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%% directory

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\comparison_experiment')

#%%

table = pd.read_excel('table.xlsx')


#%%


np.mean(table['k_SEDC'])

np.mean(table['similarity_sedc'])
np.mean(table['similarity_lime'])
np.mean(table['similarity_shap'])
np.mean(table['similarity_occlusion'])


np.median(table['mean_ct_sedc'])
np.mean(table['mean_ct_sedc'])
np.std(table['mean_ct_sedc'])

np.median(table['mean_ct_lime'])
np.mean(table['mean_ct_lime'])
np.std(table['mean_ct_lime'])

np.median(table['mean_ct_shap'])
np.mean(table['mean_ct_shap'])
np.std(table['mean_ct_shap'])

np.median(table['mean_ct_occlusion'])
np.mean(table['mean_ct_occlusion'])
np.std(table['mean_ct_occlusion'])


np.mean(table['times_counterfactual_lime'])
np.mean(table['times_counterfactual_shap'])
np.mean(table['times_counterfactual_occlusion'])


np.mean(table['times_counterfactual_lime'].loc[table['k_SEDC'] > 1])
np.mean(table['times_counterfactual_shap'].loc[table['k_SEDC'] > 1])
np.mean(table['times_counterfactual_occlusion'].loc[table['k_SEDC'] > 1])
