# Fetal-Vision: Clinical Ultrasound Inference Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![Google Cloud Run](https://img.shields.io/badge/Deployed-Google%20Cloud%20Run-4285F4.svg)](#)

> **Live Deployment:**
> **API Endpoint:**

## 🚀 Overview
Fetal-Vision is a full-stack, cloud-native computer vision pipeline designed to automate the measurement of fetal head circumference (HC) from 2D ultrasound scans. Built on the HC18 challenge dataset, this system transitions an academic deep learning model into a production-ready microservice. 

Instead of traditional segmentation, this architecture approaches HC measurement as a **5-parameter continuous regression problem** ($cx$, $cy$, $a$, $b$, $\theta$), outputting the exact geometric properties of the fetal skull.

## 📊 Clinical Performance Metrics
Evaluated on a strictly unseen 20% holdout test set (266 clinical images) without data leakage.

| Metric | Score | Clinical Context |
| :--- | :--- | :--- |
| **Intersection over Union (IoU)** | `0.765` | Strong spatial alignment exceeding standard baseline thresholds. |
| **Median Circumference Error** | `7.71 mm` | Translates to a gestational age estimation error of ~3 to 5 days. |
| **Mean Absolute Error (Center)** | `~34 px` | Calculates variance on the normalized 512x512 inference tensor. |

## 🏗️ System Architecture
The repository is engineered for high throughput, low latency, and robust error handling, reflecting modern MLOps standards.

1. **The AI Engine (`src/model.py` & `src/train.py`):** - A ResNet18 backbone modified to ingest 1-channel grayscale tensors.
   - Replaced the fully connected classification head with a sequential regression block and a Sigmoid activation to bound coordinate predictions.
   - Training pipeline incorporates dynamic learning rate scheduling (`ReduceLROnPlateau`) and geometric data augmentation (inverting bounding targets alongside horizontal tensor flips).

2. **The FastAPI Backend (`api/main.py`):**
   - Stateless REST API built for containerized deployment.
   - Includes custom, application-wide exception handling (`exceptions.py`) to gracefully reject corrupted image data, missing metadata, or mathematically improbable resolutions.

3. **In-Memory Preprocessing & Concurrency (`src/preprocess.py`):**
   - Implements a multithreaded data engine via `ThreadPoolExecutor` to handle I/O-bound image processing during training data generation.
   - Applies strict ImageNet statistics and spatial scaling based on the ultrasound's physical pixel size.

4. **Binary Caching Layer (`src/cache.py`):**
   - Generates unique SHA-256 hashes based on the raw ultrasound byte array and pixel size.
   - Bypasses redundant CPU/GPU cycles by serializing inference geometries to a `.pkl` cache, dropping cache-hit latency to `< 5ms`.

## 📂 Codebase Structure
```text
fetal-vision/
├── api/
│   └── main.py                 # FastAPI application and inference endpoints
├── data/
│   ├── cache/                  # SHA-256 hashed binary predictions
│   ├── processed/              # 512x512 .npy normalized training matrices
│   └── raw/                    # Raw HC18 challenge dataset and CSVs
├── src/
│   ├── build_labels.py         # OpenCV script to extract (cx,cy,a,b,angle) from masks
│   ├── cache.py                # Binary serialization and hashing logic
│   ├── evaluate.py             # Math engine for IoU and mm circumference calculations
│   ├── exceptions.py           # Custom exception definitions (FetalVisionError, etc.)
│   ├── model.py                # PyTorch modified ResNet18 architecture
│   ├── preprocess.py           # Multithreaded data engine and normalizer
│   ├── split_data.py           # Static 80/20 train/test holdout generation
│   └── train.py                # PyTorch training loop with LR schedulers
├── frontend/
│   └── app.py                  # Streamlit UI
├── Dockerfile                  # Container blueprint for Cloud Run
├── requirements.txt            # CPU-only PyTorch and headless OpenCV dependencies
└── fetal_vision_weights.pth    # Trained model state dictionary
```


⚙️ Local Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/PratyakshT/Fetal-Vision-Ultrasound-Triage-API.git
   cd Fetal-Vision-Ultrasound-Triage-API
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


4. **Boot the API Server:**
   ```bash
   uvicorn api.main:app --reload
   ```

The API will be available at http://localhost:8080.

5. **Launch the Frontend:**
Open a separate terminal and run:

   ```bash
   streamlit run frontend/app.py
   ```
