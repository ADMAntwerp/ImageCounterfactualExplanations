# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:56:26 2020

@author: TVermeire
"""

def sedc_target2_time(image, classifier, segments, target_class, mode, time_limit=15):
    import time
    import numpy as np
    import cv2
    
    start = time.time()
    
    result = classifier.predict(image[np.newaxis,...])
    c = np.argmax(result)
    p = result[0,target_class]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for original class
    too_long = False
    original_score = False
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
    

    
    for j in np.unique(segments):
        test_image = image.copy()
        test_image[segments == j] = perturbed_image[segments == j]
        
        result = classifier.predict(test_image[np.newaxis,...])
        c_new = np.argmax(result)
        p_new = result[0,target_class]
        
        if c_new == target_class:
            R.append([j])
            I.append(test_image)
            C.append(c_new)
            P.append(p_new)

        else: 
            sets_to_expand_on.append([j])
            P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new - result[0,c])
    
    cs = j
    
    while len(R) == 0:
        if (time.time() - start) > time_limit:
            # To create output for experiment (very dirty)
            too_long = True
            explanation = False
            perturbation = test_image
            segments_in_explanation = cs
            target_score = p_new
            new_class = c_new
            original_score = result[0,c]
            break
        
        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on,combo)
        
        for cs in combo_set: 
            
            test_image = image.copy()
            for k in cs: 
                test_image[segments == k] = perturbed_image[segments == k]
            
            result = classifier.predict(test_image[np.newaxis,...])
            c_new = np.argmax(result)
            p_new = result[0,target_class]
                
            if c_new == target_class:
                R.append(cs)
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)
            else: 
                sets_to_expand_on.append(cs)
                P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new - result[0,c])
    
    if too_long == False:         
        # Select best explanation: highest target score increase
        best_explanation = np.argmax(P - p) 
        segments_in_explanation = R[best_explanation]
        explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
        for i in R[best_explanation]:
            explanation[segments == i] = image[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation] 
        target_score = P[best_explanation]
    else: 
        print('No explanation found within time limit of ' + str(time_limit) + ' seconds.')
    
    return explanation, segments_in_explanation, perturbation, new_class, original_score, target_score, too_long                    
                