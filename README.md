# Microplastic Detection and Classification

A computer vision pipeline to detect and classify microplastics in water using YOLOv10 and a custom classification model, deployed via a Streamlit frontend.


### 1. Data Preparation
- Dataset from Roboflow (initial split: 320 train / 40 val / 40 test)
- Augmentation techniques: flip, Gaussian noise, salt & pepper noise, lighting, motion blur

### 2. Detection (YOLOv10)
- YOLOv10n trained for 50 epochs
- Results:
  - mAP50: 0.945 | mAP50-95: 0.652
  - Best class-wise mAP50: Fiber (0.993)

### 3. Custom Classifier 
- Extracted metrics: count, average size, aspect ratio
- Classified pollution level using percentile thresholds (25th, 50th, 75th, 90th)

### 4. Streamlit Frontend
- Upload image → detect microplastics → classify pollution level
- Visualizations: detection overlays, speedometer gauge, percentile bars, distribution charts

## Tech Stack
- YOLOv10 (Ultralytics), Python, OpenCV, Seaborn, Plotly, Streamlit
