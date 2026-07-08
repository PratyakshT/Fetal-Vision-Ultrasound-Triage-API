"""
Fetal-Vision Streamlit Frontend.
Provides a visual interface to communicate with the FastAPI backend 
and overlay the predicted clinical ellipse onto the ultrasound image.
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# Define the FastAPI endpoint
API_URL = "http://127.0.0.1:8000/api/v1/predict_hc"

st.set_page_config(page_title="Fetal-Vision AI", layout="wide")
st.title("Fetal-Vision: Automated Head Circumference Biometrics")
st.markdown("Upload a standard cross-sectional fetal ultrasound to predict and visualize the head circumference geometry.")

# Sidebar for inputs
with st.sidebar:
    st.header("Clinical Parameters")
    uploaded_file = st.file_uploader("Upload Ultrasound (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    pixel_size = st.number_input("Pixel Size (mm)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    predict_button = st.button("Run Inference Pipeline", type="primary")

def create_512_canvas(image_bytes, pixel_size, target_mm_per_pixel=0.1, target_shape=(512, 512)):
    """Replicates the backend's spatial normalization to map coordinates correctly."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Read in color so we can draw a colored ellipse

    scale_factor = pixel_size / target_mm_per_pixel
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, target_shape[0] - new_height)
    pad_w = max(0, target_shape[1] - new_width)
    padded_img = np.pad(
        resized_img, 
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), 
        mode='constant', constant_values=0
    )

    start_h = max(0, (new_height - target_shape[0]) // 2)
    start_w = max(0, (new_width - target_shape[1]) // 2)
    final_canvas = padded_img[start_h:start_h + target_shape[0], start_w:start_w + target_shape[1]]
    
    return final_canvas

# Main execution block
if predict_button and uploaded_file is not None:
    # 1. Read the file bytes
    file_bytes = uploaded_file.read()
    
    with st.spinner('Pinging API and running inference...'):
        # 2. Send request to your FastAPI backend
        files = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}
        data = {'pixel_size': pixel_size}
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                coords = result["data"]
                
                # 3. Create the exact normalized canvas the model saw
                canvas = create_512_canvas(file_bytes, pixel_size)
                
                # 4. Draw the predicted ellipse using OpenCV
                # cv2.ellipse expects integers for centers and axes
                center = (int(coords["cx"]), int(coords["cy"]))
                axes = (int(coords["a"]), int(coords["b"]))
                angle = coords["angle"]
                
                # Draw a bright green ellipse with a thickness of 2 pixels
                cv2.ellipse(canvas, center, axes, angle, 0, 360, (0, 255, 0), 2)
                
                # 5. Display the results in the UI
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Visualization")
                    # Convert BGR (OpenCV format) to RGB (Streamlit format)
                    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Predicted HC Ellipse (Green)", use_column_width=True)
                
                with col2:
                    st.subheader("Inference Metrics")
                    st.json(result)
                    if result.get("source") == "binary_cache":
                        st.success("⚡ Cache Hit: Bypassed GPU for zero-latency response.")
                    else:
                        st.info("🧠 Full Model Inference: Calculated via PyTorch.")
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: Ensure your FastAPI server is running. Details: {e}")