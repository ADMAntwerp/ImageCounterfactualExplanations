# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:21:51 2020

@author: TVermeire
"""

#%% Libraries

#from __future__ import absolute_import, division, print_function, unicode_literals
import time
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import os
from os import listdir
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
#from tensorflow.keras import layer

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


#%% directory

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')

#%% Explanation methods
from sedc_target2_time import sedc_target2_time

#%% Model import

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


#%% Image import

# Import function
def loadImages(path, imshape):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image).resize(imshape)
        img = np.array(img)/255.0
        # Only add image if right shape and number of channels
        if img.shape == (224,224,3):    
            loadedImages.append(img)
    return loadedImages

#%% Classes to consider
    
path = r'C:\Users\tvermeire\Dropbox\Images'
classes = listdir(path)

#%%


number_of_targets = 2
time_limit = 15
modes = ['mean', 'random', 'blur', 'inpaint']

for class_name in classes:
        
    # Import 
    path_images = 'C:/Users/tvermeire/Dropbox/Images/' + class_name + '/'
    images = loadImages(path_images,IMAGE_SHAPE)
    images = images[0:100]

    # Create directory
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_target_experiment\output')
    os.mkdir(class_name)
        
    for mode in modes: 
        
        # Experiment
        columns = ['Image']
        for i in range(1, number_of_targets+1):
            columns.append('k' + str(i))
            columns.append('s' + str(i))
            columns.append('ct' + str(i))
            columns.append('tc' + str(i))
        table = dfObj = pd.DataFrame(columns=columns)
    
        n = 1
        for image in images: 
            
            result =  classifier.predict(image[np.newaxis, ...])
            predicted_class = np.argmax(result[0], axis=-1)
            classes_sorted = np.argsort(-result)
            
            print('Classification done')
            
            # Segment image
            segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
            print('Segmentation done')
            
            k_list = []
            s_list = []
            ct_list = []
            tc_list = []
            
            for target in classes_sorted[0][1:number_of_targets+1]:
                start = time.time()
                explanation, segments_in_explanation, perturbation, new_class, original_score, target_score, too_long = sedc_target2_time(image, classifier, segments, target, mode, time_limit)
                stop = time.time()
                if too_long == False:
                    k_list.append(len(segments_in_explanation))
                    s_list.append(segments_in_explanation)
                    ct_list.append(stop-start)
                    tc_list.append(imagenet_labels[target])
                else:
                    k_list.append(np.nan)
                    s_list.append(np.nan)
                    ct_list.append(np.nan)
                    tc_list.append(imagenet_labels[target])             
            
            row = {'Image': image}
            for i in range(1, number_of_targets+1):
                row['k' + str(i)] = k_list[i-1]
                row['s' + str(i)] = s_list[i-1]
                row['ct' + str(i)] = ct_list[i-1]
                row['tc' + str(i)] = tc_list[i-1]
    
            table = table.append(row, ignore_index = True) 
            print('Image ' + str(n) + ' done')
            n += 1
            
        os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_target_experiment/output/'+ class_name)
        with pd.ExcelWriter(class_name + '_' + mode +'.xlsx') as writer: 
                table.to_excel(writer)
        os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')

    
