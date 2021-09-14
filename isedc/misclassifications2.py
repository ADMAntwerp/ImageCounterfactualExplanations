# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:58:13 2019

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

# Import cats
path_images = 'C:/Users/tvermeire/Dropbox/Images/missiles/'
images = loadImages(path_images,IMAGE_SHAPE)

#%% Predict first image

result = np.argmax(classifier.predict(images[0][np.newaxis, ...])[0], axis=-1)
print('Predicted class: ' + imagenet_labels[result])


#%% Target class

correct_class = np.argwhere(imagenet_labels == 'missile')[0][0]
print('Correct class: '+ imagenet_labels[correct_class])

#%% Image classification:  misclassificaties

labels = []
misclassifications = []
for image in images: 
    result = np.argmax(classifier.predict(image[np.newaxis, ...])[0], axis=-1)
    if result != correct_class:
        misclassifications.append(image)
        labels.append(str(imagenet_labels[result]))

#%% Images classified as class of interest
class_of_interest = np.argwhere(imagenet_labels == 'beacon')[0][0]
print('Class of interest: ' + imagenet_labels[class_of_interest])
misclassifications_focus = []
for image in misclassifications: 
    result = np.argmax(classifier.predict(image[np.newaxis, ...])[0], axis=-1)
    if result == class_of_interest:
        misclassifications_focus.append(image)

#%% Manual 
        
indices = [2,3]

misclassifications_focus= []

for i in indices:
    misclassifications_focus.append(misclassifications[i])

#%%
image = misclassifications_focus[4]
plt.imshow(image)
from sedc_t2 import sedc_t2
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
plt.imshow(mark_boundaries(image, segments))
explanation, segments_in_explanation, perturbation, new_class = sedc_t2(image, classifier, segments, correct_class, 'blur')
plt.imshow(explanation)
plt.imshow(perturbation)

#%% 
        
from seic_blur import seic_blur

originals = []
original_classes = []
explanations = []
counterfactuals = []
counterfactual_classes = []

for image in misclassifications_focus:
    
    # Classification and segmentation
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    result = classifier.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)
    
    # Explanation
    R, I, C, P = seic_blur(image, classifier, segments)
    original_score = result[0,predicted_class]
    best_explanation = np.argmax(original_score - P) 
    explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
    for i in R[best_explanation]:
        explanation[segments == i] = image[segments == i]  
    
    # Add only results in case SEIC leads to right class
    if C[best_explanation] == correct_class: 
    #if True: 
        originals.append(image)
        original_classes.append(predicted_class)
        explanations.append(explanation)
        counterfactuals.append(I[best_explanation])
        counterfactual_classes.append(C[best_explanation])
    
    
fig, axs = plt.subplots(len(originals),3,figsize=(20,5*len(originals)))        
  

for n in range(len(originals)):     
    
    # Plotting
    axs[n,0].imshow(originals[n])
    axs[n,0].set_xlabel('Class: ' + str(imagenet_labels[original_classes[n]]))
    axs[n,1].imshow(explanations[n])
    axs[n,2].imshow(counterfactuals[n])
    axs[n,2].set_xlabel('New class: ' + str(imagenet_labels[counterfactual_classes[n]]))
    

#%% save figure
os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\misclassifications')
plt.savefig('mouse_soccerball')
os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')


#%% SEIC with target 

from sedc_t2 import sedc_t2

indices = []
for i in range(10):#len(misclassifications_focus)):
    indices.append(i)

target_class = correct_class

originals = []
original_classes = []
explanations = []
counterfactuals = []
counterfactual_classes = []

for i in indices:

    image = misclassifications_focus[i]
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    result = classifier.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)
    target_score = result[0,target_class]

    explanation, segments_in_explanation, perturbation, new_class = sedc_t2(image, classifier, segments, target_class, 'blur')
    
    originals.append(image)
    original_classes.append(predicted_class)
    explanations.append(explanation)
    counterfactuals.append(perturbation)
    counterfactual_classes.append(new_class)
   

fig, axs = plt.subplots(len(originals),3,figsize=(20,5*len(originals)))          
plt.axis('Off')
for n in range(len(originals)):     
    
    # Plotting
    axs[n,0].imshow(originals[n])
    axs[n,0].set_xlabel('Class: ' + str(imagenet_labels[original_classes[n]]))
    axs[n,1].imshow(explanations[n])
    axs[n,2].imshow(counterfactuals[n])
    axs[n,2].set_xlabel('New class: ' + str(imagenet_labels[counterfactual_classes[n]]))

#%%        
os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder\misclassifications')
plt.savefig('beacon_missile_target')
os.chdir(r'C:\Users\tvermeire\Dropbox\Doctoraat\Applied Data Mining\XAI images\Spyder')


#%% Not converging

from seic_target_blur import seic_target_blur

image = misclassifications_focus[2]
result = classifier.predict(image[np.newaxis, ...])
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

target_score = result[0,correct_class]

R, I, C, P = seic_target_blur(image, classifier, segments, correct_class)

# explanation with highest target score increase
best_explanation = np.argmax(P - target_score) 
n_segments = len(R[best_explanation])

explanation_target = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
for i in R[best_explanation]:
        explanation_target[segments == i] = image[segments == i]  
        
plt.imshow(explanation_target)

#%% SEIC clarification
plt.imshow(I[best_explanation])
print('New class: ' + imagenet_labels[C[best_explanation]])


