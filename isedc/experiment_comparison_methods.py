# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:36:45 2020

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
import cv2

#%% directory

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')

#%% Explanation methods
from sedc_time import sedc_time
from explain_instance_lime import explain_instance_lime
from explain_instance_shap import explain_instance_shap
from perform_occlusion_analysis import perform_occlusion_analysis


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

#%%
    
path = r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\removal_target_experiment\output' 
classes = listdir(path)

#%% Similarity function

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


 #%% Experiment

os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/comparison_experiment')


table = dfObj = pd.DataFrame(columns=['Image', 'k_SEDC', 'similarity_sedc', 'similarity_lime', 'similarity_shap', 'similarity_occlusion', 'mean_ct_sedc', 'mean_ct_lime', 'mean_ct_shap', 'mean_ct_occlusion', 'times_counterfactual_lime', 'times_counterfactual_shap', 'times_counterfactual_occlusion'], index = [i for i in range(200)])

n = 0 #index counter
n_runs = 10
time_limit = 60
images_per_class = 10

for class_name in classes:  

    # Import 
    path_images = 'C:/Users/tvermeire/Dropbox/Images/' + class_name + '/' 
    images = loadImages(path_images,IMAGE_SHAPE)
    
    print('Class ' + class_name + ' started')
    
    images = images[300:300+images_per_class*2]
    
    counter = 0 # count images per class
    for image in images: 

        # Classify image
        result =  classifier.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(result[0], axis=-1)
        
        print('Classification done')
        
        # Segment image
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        print('Segmentation done')
        
        # Segment replacement color mean --> ADAPT ALSO BELOW
        
        #r_mean = np.mean(image[:,:,0])
        #g_mean = np.mean(image[:,:,1])
        #b_mean = np.mean(image[:,:,2])
        
        # Segment replacement method blur
        
        perturbed_image = cv2.GaussianBlur(image, (31,31), 0)
        
        # SEDC
        
        segments_in_sedc_explanations = []
        computation_times_sedc = []
        
        for i in range(n_runs):
            start = time.time()
            explanation, segments_in_explanation, perturbation, new_class, too_long = sedc_time(image, classifier, segments, 'blur', time_limit)
            stop = time.time()
            if too_long == True:
                break
            else:
                n_segments = len(segments_in_explanation)
                segments_in_sedc_explanations.append(segments_in_explanation)
                computation_times_sedc.append(stop-start)
        
        if too_long == True:
            continue
        
        print('SEDC done')
        
        # LIME
        
        segments_in_lime_explanations = [] 
        computation_times_lime = []
        counter_lime = 0
        
        for i in range(n_runs):
            start = time.time()
            explanation_lime, mask_lime = explain_instance_lime(image, classifier, n_segments)
            stop = time.time()
            segments_in_lime_explanations.append(np.unique(segments[mask_lime == 1]))
            computation_times_lime.append(stop-start)
            
            # Check whether counterfactual 
            test_image = image.copy()
            for j in np.unique(segments[mask_lime == 1]):
                test_image[segments == j] = perturbed_image[segments == j]
            
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_lime += 1
        
        if len(np.unique([len(i) for i in segments_in_lime_explanations])) > 1:
            continue
    
        print('LIME done')
        
        # SHAP
            
        segments_in_shap_explanations = []
        computation_times_shap = []
        counter_shap = 0
        
        for i in range(n_runs):
            start = time.time()
            explanation_shap, segments_in_shap_explanation = explain_instance_shap(image, classifier, segments, n_segments)
            stop = time.time()
            segments_in_shap_explanations.append(segments_in_shap_explanation)
            computation_times_shap.append(stop-start)
    
            # Check whether counterfactual 
            test_image = image.copy()
            for j in segments_in_shap_explanation:
                test_image[segments == j] = perturbed_image[segments == j]
            
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_shap += 1
        
        print('SHAP done')
        
        # Occlusion analysis
        
        segments_in_occlusion_explanations = []
        computation_times_occlusion = []
        counter_occlusion = 0
        
        for i in range(n_runs):
            start = time.time()
            explanation_occlusion, segments_in_occlusion_explanation = perform_occlusion_analysis(image, classifier, segments, n_segments)
            stop = time.time()
            segments_in_occlusion_explanations.append(segments_in_shap_explanation)
            computation_times_occlusion.append(stop-start)   
            
            # Check whether counterfactual 
            test_image = image.copy()
            for j in segments_in_occlusion_explanation:
                test_image[segments == j] = perturbed_image[segments == j]
            
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_occlusion += 1
                
        print('Occlusion done')
        
        # Calculate metrics
            
        similarity_sedc = calculate_similarity(segments_in_sedc_explanations) 
        similarity_lime = calculate_similarity(segments_in_lime_explanations)
        similarity_shap = calculate_similarity(segments_in_shap_explanations) 
        similarity_occlusion = calculate_similarity(segments_in_occlusion_explanations)
        mean_ct_sedc = np.mean(computation_times_sedc)
        mean_ct_lime = np.mean(computation_times_lime)
        mean_ct_shap = np.mean(computation_times_shap)
        mean_ct_occlusion = np.mean(computation_times_occlusion)
    
        # Put metrics in table
        
        table['Image'][n] = image
        table['k_SEDC'][n] = n_segments
        table['similarity_sedc'][n] = similarity_sedc
        table['similarity_lime'][n] = similarity_lime
        table['similarity_shap'][n] = similarity_shap
        table['similarity_occlusion'][n] = similarity_occlusion
        table['mean_ct_sedc'][n] = mean_ct_sedc
        table['mean_ct_lime'][n] = mean_ct_lime
        table['mean_ct_shap'][n] = mean_ct_shap
        table['mean_ct_occlusion'][n] = mean_ct_occlusion
        table['times_counterfactual_lime'][n] = counter_lime
        table['times_counterfactual_shap'][n] = counter_shap
        table['times_counterfactual_occlusion'][n] = counter_occlusion
        
        counter += 1
        n += 1
        print("Iteration " + str(n) + " done")


        # Output table    
        with pd.ExcelWriter('table.xlsx') as writer: 
                table.to_excel(writer)

        
        if counter == images_per_class:
            break


