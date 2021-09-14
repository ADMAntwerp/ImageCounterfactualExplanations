 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:37:06 2019

@author: TVermeire
"""

def sedc_irreducible(image, classifier, segments, mode):
    
    import numpy as np
    import itertools
    
    result = classifier.predict(image[np.newaxis,...])
    c = np.argmax(result)
    p = result[0,c]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for original class

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
    
    
    # Search explanations
    
    for j in np.unique(segments):
        test_image = image.copy()
        test_image[segments == j] = perturbed_image[segments == j]
        
        result = classifier.predict(test_image[np.newaxis,...])
        c_new = np.argmax(result)
        p_new = result[0,c]
        
        if c_new != c:
            R.append([j])
            I.append(test_image)
            C.append(c_new)
            P.append(p_new)

        else: 
            sets_to_expand_on.append([j])
            P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new)
    
    
    while len(R) == 0:
        combo = np.argmax(p - P_sets_to_expand_on)
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
            p_new = result[0,c]
                
            if c_new != c:
                R.append(cs)
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)

            else: 
                sets_to_expand_on.append(cs)
                P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new)
                
    # Select best explanation: highest score reduction
    
    best_explanation = np.argmax(p - P) 
    segments_in_explanation = R[best_explanation]
    length = len(segments_in_explanation)
    explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
    for i in R[best_explanation]:
        explanation[segments == i] = image[segments == i]
    perturbation = I[best_explanation]
    new_class = C[best_explanation]
       
    reduced = 0
    
    # Check whether irreducible
     
    if length > 2:
        
        R = [] #list of explanations
        I = [] #corresponding perturbed images
        C = [] #corresponding new classes
        P = [] #corresponding scores for original class
        
        for i in range(2,length):
            subsets = list(itertools.combinations(segments_in_explanation,i))
            for subset in subsets:
                test_image = image.copy()
                for k in subset:
                    test_image[segments == k] = perturbed_image
                
                result = classifier.predict(test_image[np.newaxis,...])
                c_new = np.argmax(result)
                p_new = result[0,c]
    
                if c_new != c:
                    R.append(subset)
                    I.append(test_image)
                    C.append(c_new)
                    P.append(p_new)
            
            if len(R) != 0:
                break

        # Adapt explanation if necessary
        if len(R) != 0:
            reduced = 1
            best_explanation = np.argmax(p - P)
            segments_in_explanation = R[best_explanation]
            explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
            for i in R[best_explanation]:
                explanation[segments == i] = image[segments == i]
            perturbation = I[best_explanation]
            new_class = C[best_explanation]
    

              
    return explanation, segments_in_explanation, perturbation, new_class, reduced                 
                
