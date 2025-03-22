import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import tempfile
import numpy as np
import pyttsx3
import os
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# Add safe globals to avoid UnpicklingError
add_safe_globals([DetectionModel])



st.set_page_config(layout="wide")
st.title("🚘 ADAS Sensor Fusion & Decision Dashboard")

# Navigation Sidebar
section = st.sidebar.selectbox("Select Section", [
    "📷 Camera Outputs",
    "📡 Radar Outputs",
    "🛰 Lidar Outputs",
    "🔀 Sensor Fusion Outputs",
    "🎯 YOLO Detections",
    "📌 Camera-Lidar Object Placement",
    "🎥 Live Object Detection",
    "🧠 ADAS Decision Making"
])

# Section 1: Camera Outputs
if section == "📷 Camera Outputs":
    st.header("Camera Preprocessing Outputs")
    camera_folder = "outputs/camera_preprocessing_annotated"
    for cam in sorted(os.listdir(camera_folder)):
        cam_path = os.path.join(camera_folder, cam)
        if os.path.isdir(cam_path):
            st.subheader(f"Camera View: {cam}")
            imgs = sorted(os.listdir(cam_path))
            for img in imgs:
                img_path = os.path.join(cam_path, img)
                st.image(img_path, use_container_width=True)

# Section 2: Radar Outputs
elif section == "📡 Radar Outputs":
    st.header("Radar CSV Outputs")
    radar_folder = "outputs/simple_radar_csv"
    radar_files = sorted(os.listdir(radar_folder))
    for file in radar_files:
        st.subheader(file)
        df = pd.read_csv(os.path.join(radar_folder, file))
        st.dataframe(df.head(200))

# Section 3: Lidar Outputs
elif section == "🛰 Lidar Outputs":
    st.header("Lidar CSV Outputs")
    lidar_folder = "outputs/simple_lidar_csv"
    lidar_files = sorted(os.listdir(lidar_folder))
    for file in lidar_files:
        st.subheader(file)
        df = pd.read_csv(os.path.join(lidar_folder, file))
        st.dataframe(df.head(200))

# Section 4: Sensor Fusion Outputs
elif section == "🔀 Sensor Fusion Outputs":
    st.header("Sensor Fused Outputs (Radar + Camera)")
    fusion_folder = "outputs/fused_output"
    fusion_files = sorted(os.listdir(fusion_folder))
    for file in fusion_files:
        st.subheader(file)
        df = pd.read_csv(os.path.join(fusion_folder, file))
        st.dataframe(df.head(200))

# Section 5: YOLO Detections
elif section == "🎯 YOLO Detections":
    st.header("YOLOv5 Object Detection Outputs")
    yolo_folder = "outputs/yolov5_detections"
    yolo_files = sorted(os.listdir(yolo_folder))
    for file in yolo_files:
        st.subheader(file)
        df = pd.read_csv(os.path.join(yolo_folder, file))
        st.dataframe(df.head(200))

# Section 6: Camera-Lidar Object Placement
elif section == "📌 Camera-Lidar Object Placement":
    st.header("Camera-Lidar Object Placement Correlation")
    placement_folder = "outputs/object_placement"
    placement_files = sorted(os.listdir(placement_folder))
    for file in placement_files:
        st.subheader(file)
        try:
            file_path = os.path.join(placement_folder, file)
            if os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path)
                if not df.empty:
                    st.dataframe(df.head(200))
                else:
                    st.warning(f"⚠️ File `{file}` is empty.")
            else:
                st.warning(f"⚠️ File `{file}` is zero bytes. Skipping.")
        except Exception as e:
            st.error(f"❌ Failed to load `{file}`: {e}")



elif section == "🎥 Live Object Detection":
    st.header("🚗 ADAS Detection - YOLOv5 + Risk Estimation System")

    # Choose Mode
    option = st.radio("Choose Detection Mode", ("📷 Live Camera", "📁 Upload Video"))
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)

    # Load YOLO Model (either method works now)
    try:
        model = YOLO("yolov5su.pt")  # If local file exists
    except Exception as e:
        st.warning("⚠ Failed to load yolov5s.pt, using default online model")
        model = YOLO("yolov5s")  # Fallback to default

    # Initialize Speech Engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    # -------------------- Detection & Decision Logic --------------------
    def detect_and_decide(frame):
        results = model.predict(frame, conf=confidence_threshold, imgsz=640)
        annotated_frame = results[0].plot()

        # Simulate obstacle distance & speed
        distance = np.random.uniform(2.0, 30.0)  # in meters
        speed = np.random.uniform(10.0, 60.0)    # in km/h

        # Calculate Time to Collision (TTC)
        ttc = distance / (speed * 1000 / 3600 + 1e-3)

        # Decision Making
        if ttc < 2:
            decision = "⚠ EMERGENCY BRAKE"
            color = (0, 0, 255)
            engine.say("Danger! Brake Immediately.")
        elif ttc < 4:
            decision = "⚠ SLOW DOWN - OBSTACLE AHEAD"
            color = (0, 165, 255)
            engine.say("Caution! Obstacle ahead.")
        else:
            decision = "✅ CLEAR - DRIVE SAFE"
            color = (0, 255, 0)
            engine.say("All clear. Drive safe.")

        engine.runAndWait()

        # Add overlays
        cv2.putText(annotated_frame, f"TTC: {ttc:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated_frame, f"Decision: {decision}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return annotated_frame

    # -------------------- Option 1: Live Camera --------------------
    if option == "📷 Live Camera":
        run_live = st.checkbox("Start Live ADAS Detection")
        if run_live:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = detect_and_decide(frame)
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated, channels="RGB")
            cap.release()

    # -------------------- Option 2: Upload Video --------------------
    elif option == "📁 Upload Video":
        uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            st.success("✅ Video Uploaded Successfully")
            st.info(f"📁 Selected File: {uploaded_file.name}")

            start_processing = st.button("▶ Start ADAS Video Detection")
            if start_processing:
                st.info("🔍 Processing Started...")

                # Save uploaded video to temp directory
                temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"📍 Video Path: {temp_video_path}")

                cap = cv2.VideoCapture(temp_video_path)
                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated = detect_and_decide(frame)
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated, channels="RGB")
                cap.release()
                st.success("✅ Video Detection Completed!")

# Section 8: ADAS Decision Making
elif section == "🧠 ADAS Decision Making":
    st.header("⚖️ Collision Risk / TTC / Decision Rules")
    decision_folder = "outputs/decision_outputs"
    decision_files = sorted(os.listdir(decision_folder))
    for file in decision_files:
        st.subheader(file)
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(decision_folder, file))
            st.dataframe(df.head(200))

            # Add Risk Heatmap Visualization if 'risk' column exists
            if 'risk' in df.columns:
                st.subheader("📊 Risk Level Heatmap")
                fig, ax = plt.subplots(figsize=(10, 4))
                risk_data = df[['risk']].T
                sns.heatmap(risk_data, cmap='coolwarm', annot=True, cbar=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

            # Add TTC Visualization if 'TTC' column exists
            if 'TTC' in df.columns:
                st.subheader("⏱ Time-To-Collision (TTC) Distribution")
                fig_ttc, ax_ttc = plt.subplots(figsize=(10, 4))
                sns.histplot(df['TTC'], bins=20, kde=True, color='orange', ax=ax_ttc)
                ax_ttc.set_title("TTC Distribution")
                ax_ttc.set_xlabel("TTC (seconds)")
                ax_ttc.set_ylabel("Frequency")
                st.pyplot(fig_ttc)

            # Add Decision Rule Explanation Visualization if 'decision' column exists
            if 'decision' in df.columns:
                st.subheader("📍 ADAS Decision Rule Analysis")
                fig_decision, ax_decision = plt.subplots(figsize=(10, 4))
                sns.countplot(data=df, x='decision', palette='Set2', ax=ax_decision)
                ax_decision.set_title("Decision Rules Distribution")
                ax_decision.set_xlabel("ADAS Decision")
                ax_decision.set_ylabel("Count")
                st.pyplot(fig_decision)

        elif file.endswith(".png"):
            st.image(os.path.join(decision_folder, file), use_container_width=True)