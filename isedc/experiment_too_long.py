# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:42:56 2020

@author: TVermeire
"""
#%%
from sedc_target2_time import sedc_target2_time

correct_class = np.argwhere(imagenet_labels == "box turtle")
image = terrapins[1]
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

start = time.time()
explanation, segments_in_explanation, perturbation, new_class, original_score, target_score, too_long = sedc_target2_time(image, classifier, segments, correct_class, 'blur')
print(time.time()-start)