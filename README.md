🤖 Computer Vision Toolbox: Segmentation and Preprocessing ToolsThis repository provides two independent Python utilities designed to streamline common tasks in computer vision workflows: image preprocessing for model training and conversion of polygon annotations to segmentation masks.🛠️ Requirements1. SoftwarePython 3.x2. LibrariesDependencies differ by module. You can install all required libraries using pip:Bashpip install opencv-python numpy Pillow chardet scikit-image concurrent.futures matplotlib
📦 Module 1: Dish Centering Pipeline (dish-centering-pipeline.py)This script implements a robust pipeline to localize, segment, and center a specific object (e.g., a dish/bowl) within a complex scene. This is ideal for generating consistent, centered training data from raw input images.⚙️ Workflow & FunctionalityThe script integrates an industrial machine vision approach (Halcon-style segmentation) with initial OpenCV template matching:Template Matching: Locates the object (bowl) using a small template image ($\text{TM\_CCOEFF\_NORMED}$ method).Cropping: Crops the image to the detected template size.Segmentation & Dilation: Uses Thresholding and Dilation to accurately isolate the object.Centering: Calculates the centroid of the segmented object and uses an Affine Transformation (cv2.warpAffine) to perfectly translate the object to the center of the cropped frame.📂 File Structure (for Module 1)/Module_1_Pipeline/
├── dish-centering-pipeline.py
├── original_images/        <-- INPUT: Raw, uncropped images
├── disher/
│   └── tempalte_image.png  <-- INPUT: Grayscale template image
└── data_unsuprevised/      <-- OUTPUT: Centered, black-background images
💡 Key Parameters (for Module 1)These constants are located at the top of the script and can be tuned for different objects:ConstantDescriptionDefault ValueBOWL_THRESHOLDPixel intensity cutoff for initial object segmentation.55DILATION_RADIUSRadius for the circular kernel used to expand the mask.30MIN_AREAMinimum required pixel area for a detected region to be considered a valid object.1300000💻 Module 2: JSON to Mask Converter (json-to-mask-converter.py)This script converts polygon annotations stored in a common JSON format (often used by labeling tools like ISAT) into binary segmentation mask images (PNG format). It utilizes multithreading for efficient bulk processing.⚙️ Workflow & FunctionalityRobust JSON Decoding: Automatically detects file encoding (chardet) and attempts multiple fallbacks to ensure file access.Polygon Rasterization: Uses skimage.draw.polygon to accurately fill the pixel area defined by the annotation coordinates.Mask Generation: Creates a single-channel (Luminosity) PNG image where annotated regions are marked as 255 (white) and the background as 0 (black).Multithreading: Leverages ThreadPoolExecutor to process numerous JSON files concurrently, minimizing total processing time.📂 File Structure (for Module 2)/Module_2_Converter/
├── json-to-mask-converter.py
├── json_datafiles/         <-- INPUT: JSON annotation files
│   ├── image1.json
│   └── image2.json
└── mask_datafiles/         <-- OUTPUT: Binary PNG mask files
    ├── image1.png
    └── image2.png
🏃 How to Run (for Module 2)The main execution block defines the input and output paths:Python# Example usage (adjust paths as needed)
json_path = r"json_datafiles"
save_path = r"mask_datafiles"
json2mask_multi(json_path, save_path, max_workers=4)
✨ Bonus: CLAHE Contrast Enhancement ExampleThe third code snippet you shared demonstrates using CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L-channel of the LAB color space to enhance image brightness and contrast without color distortion.This function (apply_clahe_color) can be easily integrated into the dish-centering-pipeline.py script as a pre-processing step right after image reading, if contrast adjustment is necessary for better template matching or segmentation.
