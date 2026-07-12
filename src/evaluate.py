import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd

# Ensure imports work from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FetalHCModel
from src.preprocess import preprocess_image

def calculate_iou(pred_params, true_params, shape=(512, 512)):
    mask_pred = np.zeros(shape, dtype=np.uint8)
    mask_true = np.zeros(shape, dtype=np.uint8)
    
    # cv2.ellipse expects: canvas, center, axes, angle, startAngle, endAngle, color, thickness
    cv2.ellipse(
        mask_pred, 
        (int(pred_params['cx']), int(pred_params['cy'])), 
        (int(pred_params['a']), int(pred_params['b'])), 
        pred_params['angle'], 0, 360, 255, -1
    )
    
    cv2.ellipse(
        mask_true, 
        (int(true_params['cx']), int(true_params['cy'])), 
        (int(true_params['a']), int(true_params['b'])), 
        true_params['angle'], 0, 360, 255, -1
    )
    
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()
    
    return intersection / union if union > 0 else 0

def calculate_circumference_mm(a, b, pixel_size):
    # Convert pixel semi-axes back to physical millimeters
    a_mm = a * pixel_size
    b_mm = b * pixel_size
    
    term1 = 3 * (a_mm + b_mm)
    term2 = np.sqrt((3 * a_mm + b_mm) * (a_mm + 3 * b_mm))
    return np.pi * (term1 - term2)

def evaluate_model(csv_path, raw_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting Evaluation Engine on: {device}")
    
    # Load Model
    model = FetalHCModel(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    df = pd.read_csv(csv_path)
    
    metrics = {
        'iou': [], 'cx_err': [], 'cy_err': [], 
        'a_err': [], 'b_err': [], 'circ_err_mm': []
    }
    
    print(f"Evaluating {len(df)} images...")
    
    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = os.path.join(raw_dir, row['filename'])
            if not os.path.exists(img_path):
                continue
                
            # Read and run through the data engine
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            pixel_size = row.get('pixel size', 0.1) # Default to 0.1 if missing for testing
            
            final_tensor, scale_factor, _, _, _, _ = preprocess_image(img, pixel_size)
            input_tensor = torch.tensor(final_tensor).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            preds = model(input_tensor)[0].cpu().numpy()
            
            # Denormalize predictions
            pred_dict = {
                'cx': preds[0] * 512.0,
                'cy': preds[1] * 512.0,
                'a': preds[2] * 512.0,
                'b': preds[3] * 512.0,
                'angle': preds[4] * 180.0
            }
            
            true_dict = {
                'cx': row['cx'], 'cy': row['cy'],
                'a': row['a'], 'b': row['b'], 'angle': row['angle']
            }
            
            # Calculate Scale-Aware Metrics
            true_circ = calculate_circumference_mm(true_dict['a'] / scale_factor, true_dict['b'] / scale_factor, pixel_size)
            pred_circ = calculate_circumference_mm(pred_dict['a'] / scale_factor, pred_dict['b'] / scale_factor, pixel_size)
            
            metrics['iou'].append(calculate_iou(pred_dict, true_dict))
            metrics['circ_err_mm'].append(abs(true_circ - pred_circ))
            metrics['cx_err'].append(abs(pred_dict['cx'] - true_dict['cx']))
            metrics['cy_err'].append(abs(pred_dict['cy'] - true_dict['cy']))
            metrics['a_err'].append(abs(pred_dict['a'] - true_dict['a']))
            metrics['b_err'].append(abs(pred_dict['b'] - true_dict['b']))

    # Print Final Report
    print("\n" + "="*40)
    print("FETAL-VISION CLINICAL METRICS REPORT")
    print("="*40)
    print(f"Mean Intersection over Union (IoU): {np.mean(metrics['iou']):.4f}")
    print(f"Median Circumference Error:       {np.median(metrics['circ_err_mm']):.2f} mm")
    print("-" * 40)
    print("Mean Absolute Error (Pixels on 512x512 tensor):")
    print(f" - Center X (cx): {np.mean(metrics['cx_err']):.2f} px")
    print(f" - Center Y (cy): {np.mean(metrics['cy_err']):.2f} px")
    print(f" - Semi-Major (a): {np.mean(metrics['a_err']):.2f} px")
    print(f" - Semi-Minor (b): {np.mean(metrics['b_err']):.2f} px")
    print("="*40)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    TEST_CSV = os.path.join(base_dir, "data", "raw", "test_split.csv") 
    TEST_DIR = os.path.join(base_dir, "data", "raw", "training_set")
    WEIGHTS = os.path.join(base_dir, "fetal_vision_weights.pth")
    
    evaluate_model(TEST_CSV, TEST_DIR, WEIGHTS)