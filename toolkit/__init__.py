import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

def log(msg):
    if debug==False:
        return
    print(time.time()-start)
    print(msg)
    
def grayscale(image_input, group=3):
    def compute_threshold_mask(gray, thresholds, index, num_levels):
        if index == 0:
            return gray <= thresholds[1]
        elif index == num_levels - 1:
            return gray > thresholds[-2]
        else:
            return (gray > thresholds[index]) & (gray <= thresholds[index + 1])
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    if img is None:
        raise ValueError(f'Failed to load image at: {image_input}')
    else:
        img = image_input.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    thresholds = np.linspace(0, 255, group + 1)
    intensity_values = np.linspace(0, 255, group)
    output_img = np.zeros_like(gray)
    
    for i in range(group):
        mask = compute_threshold_mask(gray, thresholds, i, group)
        output_img[mask] = intensity_values[i]
    
    return output_img.astype(np.uint8)
