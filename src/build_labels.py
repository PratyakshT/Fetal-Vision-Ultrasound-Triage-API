"""
Extracts target ellipse geometries (cx, cy, a, b, angle) from raw HC18 annotation masks.
Applies the exact same resizing and cropping logic used in the data engine 
to ensure ground-truth coordinates perfectly map to the 512x512 training tensors.
"""

import os
import cv2
import numpy as np
import pandas as pd

def generate_training_csv(raw_csv_path, raw_dir, output_csv_path, target_shape=(512, 512), target_mm_per_pixel=0.1):
    # Load the basic Zenodo CSV
    df = pd.read_csv(raw_csv_path)
    output_data = []
    
    print("Booting Label Extraction Engine...")

    for _, row in df.iterrows():
        filename = row['filename']
        # Extract pixel size based on the specific HC18 challenge CSV header
        pixel_size = row['pixel size'] 
        
        # HC18 challenge format appends "_Annotation" to the mask files
        annotation_filename = filename.replace('.png', '_Annotation.png')
        mask_path = os.path.join(raw_dir, annotation_filename)
        
        if not os.path.exists(mask_path):
            continue

        # 1. Read the sonographer's drawn mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Apply the EXACT same spatial transformations as src/preprocess.py
        scale_factor = pixel_size / target_mm_per_pixel
        new_width = int(mask.shape[1] * scale_factor)
        new_height = int(mask.shape[0] * scale_factor)
        
        # Use INTER_NEAREST for masks to prevent blurring the binary edges
        resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        pad_h = max(0, target_shape[0] - new_height)
        pad_w = max(0, target_shape[1] - new_width)
        padded_mask = np.pad(
            resized_mask, 
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), 
            mode='constant', constant_values=0
        )
        
        start_h = max(0, (new_height - target_shape[0]) // 2)
        start_w = max(0, (new_width - target_shape[1]) // 2)
        final_mask = padded_mask[start_h:start_h + target_shape[0], start_w:start_w + target_shape[1]]
        
        # 3. Extract the 5 geometric parameters from the standardized mask
        # Find the contour of the drawn ellipse
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
            
        # Isolate the largest drawn shape (the head circumference)
        # We use length instead of contourArea because HC18 annotations are often 1-pixel thin outlines
        largest_contour = max(contours, key=len)
        
        # OpenCV requires at least 5 points to mathematically fit an ellipse
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (axes_a, axes_b), angle = ellipse
            
            output_data.append({
                'filename': filename,
                'pixel size': float(pixel_size),  # <-- ADD THIS LINE
                'cx': float(cx),
                'cy': float(cy),
                'a': float(axes_a / 2.0), 
                'b': float(axes_b / 2.0),
                'angle': float(angle)
            })

    # 4. Save the finalized training targets
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Success. Mapped {len(output_df)} coordinate labels to {output_csv_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_CSV = os.path.join(base_dir, "data", "raw", "training_set_pixel_size_and_HC.csv")
    RAW_DIR = os.path.join(base_dir, "data", "raw", "training_set")
    OUTPUT_CSV = os.path.join(base_dir, "data", "raw", "training_labels_formatted.csv")
    
    generate_training_csv(RAW_CSV, RAW_DIR, OUTPUT_CSV)