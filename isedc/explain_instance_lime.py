# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:09:52 2019

@author: TVermeire
"""

def explain_instance_lime(image, classifier, number_features):
    
    from lime import lime_image
    
    explainer = lime_image.LimeImageExplainer()
    explanation_lime = explainer.explain_instance(image, classifier.predict, top_labels=2, hide_color=0, num_samples=1000, random_seed=42)
    explanation, mask = explanation_lime.get_image_and_mask(explanation_lime.top_labels[1], positive_only=True, num_features=number_features, hide_rest=True)
    
    return explanation, mask #seems not easy  to return segmentation and important superpixels
