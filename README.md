# Traffic Analysis Project

A comprehensive traffic congestion prediction system that combines computer vision and time series machine learning to predict congestion levels at traffic intersections.

## Overview

This project predicts traffic congestion at entry and exit points of intersections using a 5-stage pipeline:
1. **Data Ingestion & Preprocessing** - Video and metadata loading
2. **Computer Vision** - Vehicle detection and tracking
3. **Feature Engineering** - Traffic attribute extraction
4. **Time Series Analysis** - Temporal feature engineering
5. **Model Training & Prediction** - ML model training and submission generation

## Project Structure

```
├── data/
│   ├── dataset/           # Video files (normanniles1-4/)
│   ├── Train.csv         # Training metadata
│   ├── TestInputSegments.csv  # Test metadata
│   └── cv_features_*.csv # Computer vision features
├── notebooks/
│   ├── roi_setup.ipynb   # ROI configuration and visualization
│   └── data_loading.ipynb # Data preprocessing
├── src/
│   ├── computer_vision/  # CV detection and tracking
│   │   ├── detector.py
│   │   ├── tracker_ultralytics.py
│   │   └── attributes/   # Feature aggregators
│   ├── time_series/      # ML pipeline
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── feature_adapter.py
│   │   ├── feature_builder.py
│   │   ├── label_processor.py
│   │   ├── train.py
│   │   └── generate_submission.py
│   └── utils/            # Helper utilities
├── runs/detect/          # YOLO detection outputs
└── yolov8*.pt           # YOLO model weights
```

## Pipeline Stages

### 1. Data Ingestion & Preprocessing

**Input Data:**
- **Videos**: 4 camera feeds from Norman Niles intersection
- **Metadata**: Train.csv and TestInputSegments.csv with congestion labels
- **ROI Config**: config.json defining entry/exit/queue regions per camera

**Processing:**
- Extract camera IDs and timestamps from video filenames
- Configure regions of interest (ROI) for traffic analysis
- Sample frames at 2 FPS for processing efficiency

### 2. Computer Vision Pipeline

**Vehicle Detection:**
- Uses YOLOv8 models (n/nano, x/extra-large variants)
- Detects vehicles: cars, motorcycles, buses, trucks
- Configurable confidence threshold (default 0.3)

**Multi-Object Tracking:**
- ByteTrack algorithm for persistent vehicle tracking
- Maintains track IDs across video frames
- Enables vehicle-level analytics (flow, dwell time)

### 3. Feature Engineering

**Minute-Level Aggregation** (7 specialized aggregators):

| Feature Category | Metrics | Purpose |
|---|---|---|
| **Vehicle Counts** | mean/max/std vehicle counts | Traffic load and volatility |
| **Flow** | entry/exit counts, imbalance | Throughput and bottlenecks |
| **Density** | average/max density in queue | Vehicle concentration |
| **Motion** | displacement, stop ratio | Velocity proxies |
| **Dwell Time** | wait times in queue | Congestion duration |
| **Entry-Exit Delay** | system throughput time | Efficiency metrics |
| **Vehicle Composition** | bus/truck ratios | Vehicle mix impact |

**Output:** ~20 features per minute per camera

### 4. Time Series Feature Engineering

**Data Integration:**
- Merge CV features with competition metadata
- Map camera IDs to intersection names
- Ensure temporal ordering

**Temporal Features:**
- Rolling averages (5-minute windows)
- Lag features (t-1, t-2 minutes)
- Rate-of-change features
- Time-based features from video timestamps

**Target Processing:**
- Congestion classes: 0=free flowing, 1=light delay, 2=moderate delay, 3=heavy delay
- 5-minute ahead prediction horizon
- 2-minute embargo period

### 5. Model Training & Submission

**Models:**
- **LightGBM Classifier** for multiclass prediction
- Separate models for entry vs exit congestion
- Ensemble methods (Random Forest + XGBoost + LightGBM)

**Advanced Techniques:**
- Class balancing and augmentation
- Pseudo-labeling on test data
- Multiple submission strategies

## Key Technologies

- **Computer Vision**: YOLOv8, ByteTrack, OpenCV
- **Machine Learning**: LightGBM, XGBoost, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Jupyter notebooks, matplotlib

## Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure Python environment
python -m venv venv
source venv/bin/activate
```

### Run Pipeline

1. **Computer Vision Feature Extraction:**
```bash
cd src/computer_vision
python main.py  # Processes videos → features
```

2. **Time Series Training:**
```bash
cd src/time_series
python train.py  # Train models
python generate_submission.py  # Create submissions
```

### ROI Setup
- Run `notebooks/roi_setup.ipynb` to configure regions of interest
- Visualize and adjust entry/exit/queue zones per camera

## Configuration

- **config.json**: ROI definitions and camera mappings
- **src/time_series/config.py**: Model hyperparameters and target settings
- **YOLO Models**: yolov8n.pt (fast), yolov8x.pt (accurate)

## Output Files

- **Features**: `src/time_series/train_features.csv`, `test_features.csv`
- **Models**: Trained LightGBM models (saved automatically)
- **Submissions**: Multiple `submission_*.csv` files for competition

## Key Design Decisions

1. **Per-Minute Aggregation**: Reduces video complexity to manageable features
2. **ROI-Based Analysis**: Localized traffic metrics (entry/exit/queue zones)
3. **Dual Models**: Separate prediction for entry vs exit congestion patterns
4. **Temporal Context**: Rolling statistics and lag features for time series
5. **Class Imbalance Handling**: Techniques for rare heavy congestion events