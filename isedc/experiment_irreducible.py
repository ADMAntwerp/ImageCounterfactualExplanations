# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:39:00 2020

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


#%% Model import

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

#%%
    
path = r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_target_experiment\output' 
classes = listdir(path)


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


#%%

from sedc_time import sedc_time
import itertools

reduced = 0
original_size = []
reduced_size = []
reducing_too_long = 0
reducing_too_long_segments = []
counter = 0
number_per_class = 50.

for class_name  in classes:
    
    class_counter = 0
    
    # Import 
    path_images = 'C:/Users/tvermeire/Dropbox/Images/' + class_name + '/' 
    images = loadImages(path_images,IMAGE_SHAPE)
    images = images[500:500+2*number_per_class]
    
    print('Class ' + class_name + ' started')


    for image in images: 
        
        result = classifier.predict(image[np.newaxis,...])
        c = np.argmax(result)
        p = result[0,c]
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image,classifier,segments,'blur', time_limit=15)
        
        if too_long == True:
            continue
        
        counter += 1
        class_counter += 1
        
        # Check whether irreducible
        
        start = time.time()
        
        R = [] #list of explanations
        I = [] #corresponding perturbed images
        C = [] #corresponding new classes
        P = [] #corresponding scores for original class
        
        # Replacement color
        r_mean = np.mean(image[:,:,0])
        g_mean = np.mean(image[:,:,1])
        b_mean = np.mean(image[:,:,2])
        
        length = len(segments_in_explanation)
        
        if length > 2:
            
            for i in range(2,length):
                subsets = list(itertools.combinations(segments_in_explanation,i))
                for subset in subsets:
                    test_image = image.copy()
                    for k in subset:
                        test_image[segments == k] = (
                                r_mean,
                                g_mean,
                                b_mean)
            
                    result = classifier.predict(test_image[np.newaxis,...])
                    c_new = np.argmax(result)
                    p_new = result[0,c]
        
                    if c_new != c:
                        R.append(subset)
                        I.append(test_image)
                        C.append(c_new)
                        P.append(p_new)
                
                if len(R) != 0:
                    reduced += 1
                    original_size.append(len(segments_in_explanation))
                    reduced_size.append(len(R[0]))
                    print('Explanation is reduced.')
                    break 
                
                if time.time() - start > 1200:
                    reducing_too_long += 1
                    reducing_too_long_segments.append(length)
                    print('Reducing took too long: ' + str(length) + ' segments to check.')
                    break
                
        print('Iteration ' + str(counter) + ' done. This took ' + str(time.time() - start) + ' seconds.')
    
        if class_counter == number_per_class:
            break
        
        
#%%
        
relative_reducing = []
for i in range(len(original_size)):
    relative_reducing.append((original_size[i]-reduced_size[i])/original_size[i])
    
bigger_than_50 = 0
for i in range(len(relative_reducing)):
    if relative_reducing[i] > 0.5:
        bigger_than_50 += 1
        
