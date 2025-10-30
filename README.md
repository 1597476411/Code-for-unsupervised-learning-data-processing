This repository contains two primary Python scripts designed for image preprocessing and semantic segmentation dataset generation tasks, focusing on bowl-shaped objects.
1. image_processing.py 
A computer vision script to detect, segment, and center bowl-like objects within images.
🎯 Purpose
Automates the process of finding a bowl in a full-sized image, cropping the region of interest, performing segmentation to isolate the object, and finally applying an affine transformation to center the bowl in the output image. This ensures high consistency for subsequent machine learning or quality control tasks.

⚙️ Key Steps
Template Matching (template_image.png): Quickly locates the approximate bowl position.

Cropping: Extracts the region of interest based on the template match.

Segmentation: Uses a fixed threshold (BOWL_THRESHOLD) and area filtering (MIN_AREA) to identify the primary bowl contour.

Dilation: Expands the segmented area (DILATION_RADIUS) for context and mask robustness.

Centering: Calculates the centroid of the segmented region and applies an affine translation to move it to the image center.

2. json_to_mask.py
A multithreaded utility for converting polygon-based JSON annotations (from ISAT) into binary segmentation masks.
🎯 Purpose
Converts raw geometric polygon data from annotation files into black-and-white (binary) PNG mask images, which are essential for training semantic segmentation models. The process is accelerated using multithreading.

⚙️ Key Steps
JSON Decoding: Robustly reads JSON files, attempting multiple encodings (chardet + fallbacks) for compatibility.

Mask Generation: Uses skimage.draw.polygon to efficiently draw and fill the annotated polygon coordinates onto a blank NumPy array.

Multithreading: Utilizes ThreadPoolExecutor to process multiple JSON files concurrently, significantly reducing conversion time for large datasets.

PNG Output: Saves the resulting NumPy array as a black-and-white PNG mask.
