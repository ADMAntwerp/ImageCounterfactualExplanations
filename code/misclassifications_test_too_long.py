# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:11:07 2020

@author: TVermeire
"""

#%%
image = desktopcomputers[1]
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

plt.imshow(mark_boundaries(image, segments))

#%%
S = [28,22,25]
#%%

import cv2
perturbed_image = cv2.GaussianBlur(image, (31,31), 0)

perturbation = perturbed_image
for i in S:
    perturbation[segments == i] = image[segments == i]

plt.imshow(perturbation)
plt.axis('Off')

#%%
explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
for i in S:
    explanation[segments == i] = image[segments == i]
plt.imshow(explanation)
plt.axis('Off')


#%%
result = classifier.predict(perturbation[np.newaxis, ...])
predicted_class = np.argmax(result)
print('Class: ' + imagenet_labels[predicted_class])

#%%
import numpy as np
target_class = np.argwhere(imagenet_labels == 'military uniform')
from explain_instance_oc import explain_instance_oc
explanation, pixels_added = explain_instance_oc(image, classifier, segments, target_class)
