# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:46:21 2020

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
from sedc_time import sedc_time

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

time_limit = 15

for class_name in classes:
    
    # Import 
    path_images = 'C:/Users/tvermeire/Dropbox/Images/' + class_name + '/' 
    images = loadImages(path_images,IMAGE_SHAPE)
    
    images = images[0:100]


    # Create directory
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_experiment\output')
    os.mkdir(class_name)
    
    # Experiment

    table = dfObj = pd.DataFrame(columns=['Image', 'k_mean', 's_mean', 'ct_mean', 'nc_mean', 'k_blur', 's_blur', 'ct_blur', 'nc_blur', 'k_random', 's_random', 'ct_random', 'nc_random', 'k_inpaint', 's_inpaint', 'ct_inpaint', 'nc_inpaint'], index = [i for i in range(len(images))])
    
    n = 0 #index counter
    
    too_long_mean = 0
    too_long_blur = 0
    too_long_random = 0
    too_long_inpaint = 0
    
    for image in images: 
    
        # Classify image
        result =  classifier.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(result[0], axis=-1)
        
        print('Classification done')
        
        # Segment image
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        print('Segmentation done')
        
        # SEDC mean
        
        start = time.time()
        explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image, classifier, segments, 'mean', time_limit)
        stop = time.time()
        if too_long == False: 
            k_mean = len(segments_in_explanation)
            s_mean = segments_in_explanation
            ct_mean = stop-start
            nc_mean = imagenet_labels[new_class]
        else: 
            k_mean = np.nan
            s_mean = np.nan
            ct_mean = np.nan
            nc_mean = np.nan
            too_long_mean += 1
            
        print('SEDC mean done') 
        
        # SEDC blur
        
        start = time.time()
        explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image, classifier, segments, 'blur', time_limit)
        stop = time.time()
        if too_long == False: 
            k_blur = len(segments_in_explanation)
            s_blur = segments_in_explanation
            ct_blur = stop-start
            nc_blur = imagenet_labels[new_class]
        else: 
            k_blur = np.nan
            s_blur = np.nan
            ct_blur = np.nan
            nc_blur = np.nan
            too_long_blur += 1
                
        print('SEDC blur done')
        
        # SEDC random
        
        start = time.time()
        explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image, classifier, segments, 'random', time_limit)
        stop = time.time()
        if too_long == False:
            k_random = len(segments_in_explanation)
            s_random = segments_in_explanation
            ct_random = stop-start
            nc_random = imagenet_labels[new_class]
        else: 
            k_random = np.nan
            s_random = np.nan
            ct_random = np.nan
            nc_random = np.nan
            too_long_random += 1
            
        print('SEDC random done')
        
        # SEDC inpaint
        
        start = time.time()
        explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image, classifier, segments, 'inpaint', time_limit)
        stop = time.time()
        if too_long == False: 
            k_inpaint = len(segments_in_explanation)
            s_inpaint = segments_in_explanation
            ct_inpaint = stop-start 
            nc_inpaint = imagenet_labels[new_class]
        else: 
            k_inpaint = np.nan
            s_inpaint = np.nan
            ct_inpaint = np.nan
            nc_inpaint = np.nan
            too_long_inpaint += 1
        
        print('SEDC inpaint done')
    
        # Put metrics in table
        
        table['Image'][n] = image
        table['k_mean'][n] = k_mean
        table['s_mean'][n] = s_mean
        table['ct_mean'][n] = ct_mean
        table['nc_mean'][n] = nc_mean
        table['k_blur'][n] = k_blur
        table['s_blur'][n] = s_blur
        table['ct_blur'][n] = ct_blur
        table['nc_blur'][n] = nc_blur
        table['k_random'][n] = k_random
        table['s_random'][n] = s_random
        table['ct_random'][n] = ct_random
        table['nc_random'][n] = nc_random
        table['k_inpaint'][n] = k_inpaint
        table['s_inpaint'][n] = s_inpaint
        table['ct_inpaint'][n] = ct_inpaint
        table['nc_inpaint'][n] = nc_inpaint
    
        n += 1
        print("Image " + str(n) + " done")

    # Output table    
    os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/removal_experiment/output/'+ class_name)
    with pd.ExcelWriter(class_name + '.xlsx') as writer: 
            table.to_excel(writer)
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')
    
    
