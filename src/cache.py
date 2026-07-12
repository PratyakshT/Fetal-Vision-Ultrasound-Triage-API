import os
import pickle
import hashlib
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')

def _generate_hash(image_array, pixel_size):
    # Generates a unique SHA-256 hash for a given NumPy array and pixel_size.
    # Convert array and pixel_size to bytes to create a consistent, immutable hash
    array_bytes = image_array.tobytes()
    pixel_bytes = str(pixel_size).encode('utf-8')
    return hashlib.sha256(array_bytes + pixel_bytes).hexdigest()

def get_cached_prediction(image_array, pixel_size):
    # Checks if a prediction for this specific image already exists in the cache.
    # Returns the dictionary of parameters if found, else None.
    os.makedirs(CACHE_DIR, exist_ok=True)
    img_hash = _generate_hash(image_array, pixel_size)
    cache_path = os.path.join(CACHE_DIR, f"{img_hash}.pkl")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                prediction_data = pickle.load(f)
            return prediction_data
        except Exception as e:
            # If the binary file is corrupted, return None to force a fresh inference
            print(f"Cache read failed for {img_hash}: {e}")
            return None
    return None

def set_cached_prediction(image_array, pixel_size, prediction_dict):
    # Serializes and saves the model's prediction to the cache directory.
    # prediction_dict format: {'cx': float, 'cy': float, 'a': float, 'b': float, 'angle': float}

    os.makedirs(CACHE_DIR, exist_ok=True)
    img_hash = _generate_hash(image_array, pixel_size)
    cache_path = os.path.join(CACHE_DIR, f"{img_hash}.pkl")
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(prediction_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(f"Failed to write cache for {img_hash}: {e}")
        return False