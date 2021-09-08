# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:00:27 2019

@author: TVermeire
"""

def explain_instance_shap(image, classifier, segments, number_features):
    import shap
    import numpy as np
    
    # define a function that depends on a binary mask representing if an image region is hidden
    def mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0,1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i,:,:,:] = image
            for j in range(zs.shape[1]):
                if zs[i,j] == 0:
                    out[i][segmentation == j,:] = background
        return out
    def f(z):
        return classifier.predict(mask_image(z, segments, image))
    
    # use Kernel SHAP to explain the network's predictions
    n_segments = len(np.unique(segments))
    explainer = shap.KernelExplainer(f, np.zeros((1,n_segments)))
    shap_values = explainer.shap_values(np.ones((1,n_segments)), nsamples=1000)  
    
    relevant_shap = shap_values[np.argmax(classifier.predict(image[np.newaxis, ...])[0], axis = -1)]
    ranked_segments = np.argsort(-relevant_shap)

    explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
    for i in ranked_segments[0,0:number_features]:
        explanation[segments == i] = image[segments == i]
        
    segments_in_explanation = ranked_segments[0,0:number_features]
    return explanation, segments_in_explanation

