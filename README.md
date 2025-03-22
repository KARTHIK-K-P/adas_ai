# ADAS & AI/ML Assignment - Obstacle Detection & Sensor Fusion

## 📌 Objective
Brief about the assignment goals, importance of ADAS, sensor fusion, AI in modern vehicles.

## 🔁 System Architecture
- ADAS Architecture Flowchart (attach `C:\Users\karth\Downloads\ADAS_Complete_Project_Final (1)\ADAS_Flowchart.drawio(1).png`)
- Sensors used: Camera, Radar, Lidar
- ML Models integrated

## 📂 Dataset Used
- Camera images: ["C:\Users\karth\Downloads\ADAS_Complete_Project_Final (1)\outputs\camera_preprocessing_annotated"]
- Radar data: "outputs\simple_radar_csv"
- Lidar data: ["outputs\simple_lidar_csv"]
- Number of samples and structure

## ⚙️ Data Preprocessing
- Camera preprocessing using YOLOv5 or Transfer Learning (MobileNet, EfficientNet, etc.)
- Radar preprocessing (distance, angle, speed, field of view)
- Object vector generation
- Timestamp alignment

## 🤖 Object Detection & Placement
- Camera-Lidar correlation logic
- Contour analysis and object placement
- Obstruction detection using custom ML models

## 🧠 Decision Making Logic
- TTC (Time-To-Collision) calculation
- Risk classification and visual indicators
- Braking and lane assist logic

## 📈 Visualizations
- Sensor Fusion overlays
- Risk level heatmaps
- Decision logic plots

## 🛠 Improvements Suggested
1. Real-time fusion pipeline with faster inference models
2. Enhance radar preprocessing with dynamic Doppler filtering
3. Add stereo vision or depth estimation module

## 📚 References
Datasets:
- nuScenes Dataset: https://www.nuscenes.org/download
- KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
- Waymo Open Dataset: https://waymo.com/open/
- BDD100K Dataset: https://bdd-data.berkeley.edu/

Models:
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8: https://github.com/ultralytics/ultralytics
- MobileNetV2: https://keras.io/api/applications/mobilenet/
- EfficientNet: https://keras.io/api/applications/efficientnet/

Tools & Libraries:
- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- Plotly: https://plotly.com/
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- PyTorch: https://pytorch.org/


hugging face space : https://huggingface.co/spaces/karthikkp11/moonrider
git hub link : https://github.com/KARTHIK-K-P/adas_ai