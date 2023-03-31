# -*- coding:UTF-8 -*-
import numpy as np
import os
from skimage import io
import tqdm

dir = '.\out'

prediction_path=sorted([
    os.path.join(dir, fname)
    for fname in os.listdir(dir)
    if fname.endswith("_Mine.png")])


groundtruth_path=sorted([
    os.path.join(dir, fname)
    for fname in os.listdir(dir)
    if fname.endswith("_Segmentation.png")])


jaccard_scores = []
dice_scores = []
pixel_accuracy = []
sensitivity = []
specitivity = []
 
predictions_formatted = []
ground_truth_formatted = []
    
for i in tqdm.tqdm(range(0, len(prediction_path))):
    # Adapt prediction and ground truth
    prediction = io.imread(prediction_path[i])
    ground_truth = io.imread(groundtruth_path[i])
    
    predictions_formatted.append(prediction.flatten())
    ground_truth_formatted.append(ground_truth.flatten())
    number_of_pixels = 512*512  
    
    # prediction = np.argmax(prediction_path[i], axis=-1)
    # prediction = np.expand_dims(prediction, axis=-1)
    # predictions_formatted.append(prediction.flatten())
    # ground_truth = np.array(load_img(test_target_img_paths[i], target_size=img_size, color_mode="grayscale"))
    # ground_truth = (np.expand_dims(ground_truth, 2)/255).astype(int)
    # ground_truth_formatted.append(ground_truth.flatten())
    # number_of_pixels = img_size[0]*img_size[1]    
    
    # Get Jaccard score
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    jaccard_scores.append(np.sum(intersection) / np.sum(union))  
    
    # Get Dice coefficient
    intersection = np.sum(ground_truth.flatten() == prediction.flatten())
    dice_scores.append((2 * np.sum(intersection) ) / (number_of_pixels + number_of_pixels))
    
    # Pixel-based metrics
    equal_pixels = 0
    number_of_true_positives = 0
    number_of_true_negatives = 0
    number_of_false_negatives = 0
    number_of_false_positives = 0
    
    for row in range(len(ground_truth)):
        for column in range(len(ground_truth[row])):
            
            if prediction[row][column] == ground_truth[row][column]:
                equal_pixels += 1
                
            if ground_truth[row][column] == 1 and prediction[row][column] == ground_truth[row][column]:
                number_of_true_positives += 1
                
            if ground_truth[row][column] == 0 and prediction[row][column] == ground_truth[row][column]:
                number_of_true_negatives +=1
            
            if prediction[row][column] == 1 and prediction[row][column] != ground_truth[row][column]:
                number_of_false_positives += 1
            
            if prediction[row][column] == 0 and prediction[row][column] != ground_truth[row][column]:
                number_of_false_negatives += 1
                
    # Pixel accuracy: (Correct predictions / Number of predictions)      
    pixel_accuracy.append(equal_pixels / number_of_pixels)
    # Sensitivity - Recall: True positive rate (True positives / True positives + False negatives). How many of the positives are correct
    try:sensitivity.append(number_of_true_positives / (number_of_true_positives + number_of_false_negatives))
    except:sensitivity.append(0)
    # Specitivity - True negative rate (True negative / True negative + False positives). How many of the negatives are correct
    try: specitivity.append(number_of_true_negatives / (number_of_true_negatives + number_of_false_positives))
    except: specitivity.append(0)
        
print(f"Jaccard Score: {np.mean(jaccard_scores)} \nDice Score: {np.mean(dice_scores)} \nPixel accuracy: {np.mean(pixel_accuracy)} \nSensitivity: {np.mean(sensitivity)} \nSpecificity: {np.mean(specitivity)}")