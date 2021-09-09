# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:49:20 2020

@author: TVermeire
"""

import numpy as np
import cv2
from time import time

# BEST-FIRST: difference between target class score and predicted class score

def sedc_t2_fast(image, classifier, segments, target_class, mode, max_time=600):

    init_time = time()

    result = classifier.predict(image[np.newaxis, ...])

    c = np.argmax(result)
    p = result[0, target_class]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for target class
    sets_to_expand_on = []
    P_sets_to_expand_on = np.array([])

    if mode == 'mean':
        perturbed_image = np.zeros((224,224,3))
        perturbed_image[:,:,0] = np.mean(image[:,:,0])
        perturbed_image[:,:,1] = np.mean(image[:,:,1])
        perturbed_image[:,:,2] = np.mean(image[:,:,2])
    elif mode == 'blur':
        perturbed_image = cv2.GaussianBlur(image, (31,31), 0)
    elif mode == 'random':
        perturbed_image = np.random.random((224,224,3))
    elif mode == 'inpaint':
        perturbed_image = np.zeros((224,224,3))
        for j in np.unique(segments):
            image_absolute = (image*255).astype('uint8')
            mask = np.full([image_absolute.shape[0],image_absolute.shape[1]],0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            perturbed_image[segments == j] = image_segment_inpainted[segments == j]/255.0

    cf_candidates = []
    for j in np.unique(segments):
        test_image = image.copy()
        test_image[segments == j] = perturbed_image[segments == j]

        cf_candidates.append(test_image[np.newaxis,...][0])

    cf_candidates = np.array(cf_candidates)

    results = classifier.predict(cf_candidates)
    c_new_list = np.argmax(results, axis=1)
    p_new_list = results[:, target_class]

    if target_class in c_new_list:
        R = np.where(c_new_list == target_class)[0]
        I = cf_candidates[R]
        C = c_new_list[R]
        P = p_new_list[R]

    sets_to_expand_on = [[x] for x in np.where(c_new_list != target_class)[0]]
    P_sets_to_expand_on = p_new_list[np.where(c_new_list != target_class)[0]]-results[np.where(c_new_list != target_class)[0], c]

    combo_set = [0]
    
    while len(R) == 0 and len(combo_set) > 0 and max_time > time() - init_time:

        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on, combo)

        cf_candidates = []

        for cs in combo_set:
            test_image = image.copy()
            for k in cs:
                test_image[segments == k] = perturbed_image[segments == k]

            cf_candidates.append(test_image)
        cf_candidates = np.array(cf_candidates)

        results = classifier.predict(cf_candidates)
        c_new_list = np.argmax(results, axis=1)
        p_new_list = results[:, target_class]

        if target_class in c_new_list:
            selected_idx = np.where(c_new_list == target_class)[0]

            R = np.array(combo_set)[selected_idx].tolist()
            I = cf_candidates[selected_idx]
            C = c_new_list[selected_idx]
            P = p_new_list[selected_idx]

        sets_to_expand_on += np.array(combo_set)[np.where(c_new_list != target_class)[0]].tolist()
        P_sets_to_expand_on = np.append(P_sets_to_expand_on, p_new_list[np.where(c_new_list != target_class)[0]] - results[np.where(c_new_list != target_class)[0], c])

    # Select best explanation: highest target score increase

    if len(R) > 0:
        best_explanation = np.argmax(P - p)
        segments_in_explanation = R[best_explanation]
        explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
        for i in R[best_explanation]:
            explanation[segments == i] = image[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]


        return explanation, segments_in_explanation, perturbation, new_class


    print('No CF found on the requested parameters')
    return None, None, None, c
