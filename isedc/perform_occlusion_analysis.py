# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:20:08 2020

@author: TVermeire
"""

def perform_occlusion_analysis(image, classifier, segments, n_segments):
    
    import numpy as np
    
    result = classifier.predict(image[np.newaxis,...])
    c = np.argmax(result)
    p = result[0,c]
    P = np.array([]) #corresponding scores for original class

    perturbed_image = np.zeros((224,224,3))
    perturbed_image[:,:,0] = np.mean(image[:,:,0])
    perturbed_image[:,:,1] = np.mean(image[:,:,1])
    perturbed_image[:,:,2] = np.mean(image[:,:,2])
    
    applied_segments = np.unique(segments)

    for j in applied_segments:
        test_image = image.copy()
        test_image[segments == j] = perturbed_image[segments == j]
    
        result = classifier.predict(test_image[np.newaxis,...])
        p_new = result[0,c]  
        P = np.append(P, p-p_new)
     
    P_sorted = np.argsort(-P)
    segments_ranking = applied_segments[P_sorted]
    
    segments_in_explanation = segments_ranking[0:n_segments]
    
    explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
    for i in segments_in_explanation:
        explanation[segments == i] = image[segments == i]
        
    return explanation, segments_in_explanation