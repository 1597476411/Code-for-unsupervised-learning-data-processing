import cv2
import numpy as np
import os
import time
import shutil

original_input_folder = r"original_images"
# Final output folder (centered bowls)
final_output_folder = r"data_unsuprevised"

# Target size from Halcon logic (for reference)
TargetWidth = 1512
TargetHeight = 1512

template_path = r"disher\tempalte_image.png"
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if template is None:
    raise ValueError("Template image failed to load. Check the path.")

os.makedirs(final_output_folder, exist_ok=True)

BOWL_THRESHOLD = 55
DILATION_RADIUS = 30
MIN_AREA = 1300000

print("--- Starting Processing: Template Matching, Crop, Segmentation, Centering ---")

# Process all images in the input folder
for filename in os.listdir(original_input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(original_input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        full_process_start_time = time.time()

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Template matching (TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = template.shape

        # Crop the bowl region (in memory)
        bowl_crop = image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        gray_bowl = cv2.cvtColor(bowl_crop, cv2.COLOR_BGR2GRAY)
        H, W = gray_bowl.shape

        _, region_mask = cv2.threshold(gray_bowl, BOWL_THRESHOLD, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_AREA and area > max_area:
                max_area = area
                valid_contour = contour

        if valid_contour is None:
            print(f"Processing failed for {filename}: No valid bowl region found (area < {MIN_AREA}). Skipping save.")
            continue

        # Dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATION_RADIUS + 1, 2 * DILATION_RADIUS + 1))

        bowl_region = np.zeros_like(gray_bowl, dtype=np.uint8)
        cv2.drawContours(bowl_region, [valid_contour], 0, 255, -1)

        region_dilation = cv2.dilate(bowl_region, kernel, iterations=1)

        # Reduce domain (set background outside dilation to black)
        image_reduced = cv2.bitwise_and(bowl_crop, bowl_crop, mask=region_dilation)

        # Calculate centroid for centering
        moments = cv2.moments(region_dilation)

        if moments["m00"] == 0:
            print(f"Processing failed for {filename}: Dilation area is zero. Skipping save.")
            continue

        RegionCenterY = moments["m01"] / moments["m00"]
        RegionCenterX = moments["m10"] / moments["m00"]

        ImageCenterX = W / 2
        ImageCenterY = H / 2

        TranslateX = ImageCenterX - RegionCenterX
        TranslateY = ImageCenterY - RegionCenterY

        # Create translation matrix
        M = np.float32([[1, 0, TranslateX], [0, 1, TranslateY]])

        # Apply affine transformation (Centering)
        centered_image = cv2.warpAffine(image_reduced, M, (W, H),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        output_filename = 'ok_' + filename
        output_path = os.path.join(final_output_folder, output_filename)
        cv2.imwrite(output_path, centered_image)

        full_process_end_time = time.time()
        full_elapsed_time = (full_process_end_time - full_process_start_time) * 1000

        print(f"Processing complete: {filename} -> {output_path}, Total time: {full_elapsed_time:.4f} ms")

print("\nAll images processed, results saved.")

intermediate_folder = r"output"
if os.path.exists(intermediate_folder):
    print(f"\nCleaning intermediate folder: {intermediate_folder}")
    try:
        pass
    except Exception as e:
        print(f"Could not delete intermediate folder: {e}")
