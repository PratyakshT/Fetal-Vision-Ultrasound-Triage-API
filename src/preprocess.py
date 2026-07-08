import sys
import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the root directory is accessible for imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exceptions import CorruptedImageError, MetadataMissingError

def preprocess_image(img, pixel_size, target_mm_per_pixel=0.1, target_shape=(512, 512)):
    """
    Scales, crops, pads, and normalizes a single ultrasound image matrix based on its physical pixel size.
    """
    scale_factor = pixel_size / target_mm_per_pixel
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, target_shape[0] - new_height)
    pad_w = max(0, target_shape[1] - new_width)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded_img = np.pad(
        resized_img, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', 
        constant_values=0
    )

    start_h = max(0, (new_height - target_shape[0]) // 2)
    start_w = max(0, (new_width - target_shape[1]) // 2)
    final_tensor = padded_img[start_h:start_h + target_shape[0], start_w:start_w + target_shape[1]]

    # Normalize pixel intensities [0, 1]
    final_tensor = final_tensor.astype(np.float32) / 255.0
    
    # ImageNet Grayscale Normalization
    mean = 0.449
    std = 0.226
    final_tensor = (final_tensor - mean) / std

    return final_tensor, scale_factor, pad_top, pad_left, start_h, start_w

def process_single_ultrasound(filename, pixel_size, raw_dir, processed_dir, target_mm_per_pixel=0.1, target_shape=(512, 512)):
    """
    Reads, scales, and normalizes a single ultrasound image based on its physical pixel size.
    """
    img_path = os.path.join(raw_dir, filename)
    
    # 1. Read Image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise CorruptedImageError(filename)
    if pd.isna(pixel_size) or pixel_size <= 0:
        raise MetadataMissingError(filename)

    final_tensor, _, _, _, _, _ = preprocess_image(img, pixel_size, target_mm_per_pixel, target_shape)

    # Save as a serialized numpy array to bypass heavy image decoders during training
    save_path = os.path.join(processed_dir, filename.replace('.png', '.npy'))
    np.save(save_path, final_tensor)
    
    return filename, True

def build_data_engine(labels_csv_path, raw_dir, processed_dir, max_workers=8):
    """
    Executes the preprocessing pipeline concurrently using multithreading.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Assume labels dataset has standard HC18 challenge format
    df = pd.read_csv(labels_csv_path)
    
    print(f"Booting Data Engine: Processing {len(df)} ultrasounds using {max_workers} threads...")
    
    success_count = 0
    failure_logs = []

    # Implementing ThreadPoolExecutor to handle I/O bound image processing concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to threads
        futures = {
            executor.submit(
                process_single_ultrasound, 
                row['filename'], 
                row['pixel size'], # Requires extraction from original dataset metadata
                raw_dir, 
                processed_dir
            ): row['filename'] for _, row in df.iterrows()
        }

        # Harvest results as they complete
        for future in as_completed(futures):
            filename = futures[future]
            try:
                _, success = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                failure_logs.append(f"Failed {filename}: {str(e)}")

    print(f"Engine Shutdown. Successfully normalized {success_count}/{len(df)} images.")
    if failure_logs:
        print("Error Logs:")
        for log in failure_logs[:5]: # Print first 5 errors
            print(f" - {log}")

if __name__ == "__main__":
    # Standard entry point for independent testing
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_data_engine(
        os.path.join(base_dir, "data", "raw", "training_set_pixel_size_and_HC.csv"), 
        os.path.join(base_dir, "data", "raw", "training_set"), 
        os.path.join(base_dir, "data", "processed")
    )