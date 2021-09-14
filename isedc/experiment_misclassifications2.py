# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:30:06 2020

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

import tensorflow as tf
import tensorflow_hub as hub
#from tensorflow.keras import layer

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import pandas as pd

#%% directory

os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')


#%% Model import

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

# INCEPTION 
#classifier_url = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
#IMAGE_SHAPE = (229, 229)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

#%% Import function

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
    
path = r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\misclassifications_experiment\blur replacement'
classes = listdir(path)


#%% Number of misclassification classes to consider

number_of_classes = 5

#%% Experiment

from sedc_target2_time import sedc_target2_time

for class_name in classes: 
            
    # Import 
    path_images = 'C:/Users/tvermeire/Dropbox/Images/' + class_name +'s/'
    images = loadImages(path_images,IMAGE_SHAPE)
    
    # Create directory
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\misclassifications_experiment\blur replacement 2')
    os.mkdir(class_name)
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')
    
    # Correct class
    correct_class = np.argwhere(imagenet_labels == class_name)[0][0]
    
    # Misclassifications
    
    labels = []
    misclassifications = []
    for image in images: 
        result = np.argmax(classifier.predict(image[np.newaxis, ...])[0], axis=-1)
        if result != correct_class:
            misclassifications.append(image)
            labels.append(str(imagenet_labels[result]))
    
    # Table for misclassifications
    table_misclassifications = dfObj = pd.DataFrame(columns=['Correct class','# images','# misclassifications'])
    table_misclassifications = table_misclassifications.append({'Correct class': class_name, '# images': len(images), '# misclassifications': len(misclassifications)}, ignore_index=True)                                                         

    # Order labels based on occurrence
    ordered = pd.Series(labels).value_counts()
    labels_ordered = ordered.index.values.tolist()
    
    # Search for explanations
    table_explanations = dfObj = pd.DataFrame(columns=['Class','# misclassified','# target found'])
                                                       
    # Table too long
    table_too_long = dfObj = pd.DataFrame(columns=['original class', '# segments', '# perturbed segments', 'target score change', 'original class score change',  'image', 'perturbed image', 'current class'])
    
    for label in labels_ordered[0:number_of_classes]:
        
        class_of_interest = np.argwhere(imagenet_labels == label)[0][0]
        
        misclassifications_focus = []
        for image in misclassifications: 
            result = np.argmax(classifier.predict(image[np.newaxis, ...]))
            if result == class_of_interest:
                misclassifications_focus.append(image)
        
        originals = []
        original_classes = []
        explanations = []
        counterfactuals = []
        counterfactual_classes = []
        too_long_counter = 0
        images_too_long = []
        
        for image in misclassifications_focus:
            segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
            result = classifier.predict(image[np.newaxis, ...])
            predicted_class = np.argmax(result[0])
            target_score = result[0,correct_class]
            original_score = result[0,predicted_class]
            
            explanation, segments_in_explanation, perturbation, new_class, o_s, t_s, too_long = sedc_target2_time(image, classifier, segments, correct_class, 'blur')
            
            if too_long == False: 
                # explanation with highest target score increase                
                originals.append(image)
                original_classes.append(predicted_class)
                explanations.append(explanation)
                counterfactuals.append(perturbation)
                counterfactual_classes.append(new_class)
            
            else: 
                table_too_long = table_too_long.append({'original class': label, '# segments': len(np.unique(segments)), '# perturbed segments': len(segments_in_explanation), 'target score change': t_s-target_score, 'original class score change': o_s-original_score ,'image': image, 'perturbed image': perturbation, 'current class': imagenet_labels[new_class]}, ignore_index=True)
                images_too_long.append(perturbation)
                too_long_counter += 1
        
        # Create table explanations
        table_explanations = table_explanations.append({'Class': label, '# misclassified': len(misclassifications_focus), '# target found': len(misclassifications_focus)-too_long_counter}, ignore_index=True)
        
        # Create figure
        
        if len(originals) != 0:    
            fig, axs = plt.subplots(len(originals),3,figsize=(20,5*len(originals)))   
            
            if len(originals) == 1:
                axs[0].imshow(originals[0])
                axs[0].set_xlabel('Class: ' + str(imagenet_labels[original_classes[0]]))
                axs[1].imshow(explanations[0])
                axs[2].imshow(counterfactuals[0])
                axs[2].set_xlabel('New class: ' + str(imagenet_labels[counterfactual_classes[0]]))            
            else: 
                for n in range(len(originals)):     
                    # Plotting
                    axs[n,0].imshow(originals[n])
                    axs[n,0].set_xlabel('Class: ' + str(imagenet_labels[original_classes[n]]))
                    axs[n,1].imshow(explanations[n])
                    axs[n,2].imshow(counterfactuals[n])
                    axs[n,2].set_xlabel('New class: ' + str(imagenet_labels[counterfactual_classes[n]]))
            
        # Output figure
        os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/misclassifications_experiment/blur replacement 2/'+ class_name)
        plt.savefig(imagenet_labels[correct_class]+'_'+imagenet_labels[class_of_interest])
        os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')
    
    # Output table    
    os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/misclassifications_experiment/blur replacement 2/'+ class_name)
    with pd.ExcelWriter(class_name + '.xlsx') as writer: 
            table_misclassifications.to_excel(writer, sheet_name = 'Sheet1')
            table_explanations.to_excel(writer, sheet_name = 'Sheet2')
            table_too_long.to_excel(writer, sheet_name = 'Sheet3')
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')
       
    # Create figure too long
    if len(table_too_long) != 0:    
        fig, axs = plt.subplots(len(table_too_long),2,figsize=(20,5*len(table_too_long)))   
            
        if len(table_too_long) == 1:
            axs[0].imshow(table_too_long['image'][0])
            axs[0].set_xlabel('Class: ' + str(table_too_long['original class'][0]))
            axs[1].imshow(table_too_long['perturbed image'][0])
            axs[1].set_xlabel('New class: ' + table_too_long['current class'][0])            
        else: 
            for n in range(len(table_too_long)):     
                # Plotting
                axs[n,0].imshow(table_too_long['image'][n])
                axs[n,0].set_xlabel('Class: ' + str(table_too_long['original class'][n]))
                axs[n,1].imshow(table_too_long['perturbed image'][n])
                axs[n,1].set_xlabel('Current class: ' + table_too_long['current class'][n])
     
    # Output figure too long
    os.chdir('C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/misclassifications_experiment/blur replacement 2/'+ class_name)
    plt.savefig(imagenet_labels[correct_class]+'_too_long')
    plt.close()
    os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')    

              
        