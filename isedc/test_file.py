# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:50:06 2019

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

#INCEPTION 
#classifier_url = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
#IMAGE_SHAPE = (229, 229)


classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,), trainable=True)
])


labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


#%% Image import

#Make sure the relevant image is located in your working directory
image = Image.open("img/chihuahua.jpg")
image = image.resize(IMAGE_SHAPE)
image = np.array(image)/255.0
image = image[:,:,0:3]
plt.imshow(image)

#%% Image classification
result = classifier.predict(image[np.newaxis, ...])
predicted_class = np.argmax(result)
print('Class: ' + imagenet_labels[predicted_class])

#%% 

scores_sorted = np.argsort(-result)

labels_sorted = imagenet_labels[scores_sorted]


#%% QUICKSHIFT SEGMENTATION
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

plt.imshow(mark_boundaries(image, segments))
print("Number of superpixels: " + str(len(np.unique(segments))))

#%% alternative SEGMENTATION

segments = slic(image, n_segments=50, compactness=10, sigma=1)

from create_squared_segmentation import create_squared_segmentation

segments = create_squared_segmentation(8,8,28)
plt.imshow(mark_boundaries(image, segments))
 
#%%

from sedc import sedc
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
explanation, segments_in_explanation, perturbation, new_class = sedc(image, classifier, segments, 'inpaint')
plt.imshow(explanation)
print(imagenet_labels[new_class])


#%%
target = np.argwhere(imagenet_labels == 'suit')
from sedc_t2 import sedc_t2
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
start=time.time()
explanation, segments_in_explanation, perturbation, new_class = sedc_t2(image, classifier, segments, target, 'blur')
print(time.time()-start)
plt.imshow(perturbation)
plt.axis('off')



#%% Image import function

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

# Import images
path_images = 'C:/Users/tvermeire/Dropbox/Doctoraat/Applied Data Mining/XAI images/Spyder/img/military uniforms/'
images = loadImages(path_images,IMAGE_SHAPE)



#%% LIME
from explain_instance_lime import explain_instance_lime

start = time.time()

explanation_lime, mask_lime = explain_instance_lime(image, classifier, 1)

stop = time.time()
print('Time elapsed: ' + str(stop-start))

plt.imshow(explanation_lime)
plt.axis('Off')
#%% SHAP

from explain_instance_shap import explain_instance_shap

start = time.time()

explanation_shap, segments_in_shap_explanation = explain_instance_shap(image, classifier, segments, 1)

stop = time.time()
print('Time elapsed: ' + str(stop-start))

plt.imshow(explanation_shap)
plt.axis('Off')
#%% Plot explanations

fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].imshow(explanation_seic)
ax[1].imshow(explanation_lime)
ax[2].imshow(explanation_shap)
ax[0].set_title('SEIC')
ax[1].set_title('LIME')
ax[2].set_title('SHAP')
plt.show()




