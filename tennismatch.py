import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import os
import time

# Fix path compatibility for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model_path = 'best.pt'
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

# Streamlit page configuration
st.set_page_config(page_title="Tennis Game Tracking", layout="wide")

# Title of the application
st.title("Tennis Game Tracking")

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None

# Sidebar for upload, process, and download buttons
st.sidebar.title("Menu")
upload_button = st.sidebar.button("Upload Video")
process_button = st.sidebar.button("Process Video")
download_button = st.sidebar.button("Download Video")

# File uploader in the sidebar
uploaded_video = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# Handle video upload
if upload_button:
    if uploaded_video:
        st.session_state.uploaded_file = uploaded_video
        st.video(uploaded_video)
        st.success("Video uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a video file.")

# Handle video processing
if process_button:
    if st.session_state.uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(st.session_state.uploaded_file.read())
            temp_video_path = temp_video.name
        
        cap = cv2.VideoCapture(temp_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
            output_video_path = output_temp.name
        st.session_state.output_path = output_video_path
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        st.subheader("Processing Video... Please wait.")
        progress_bar = st.progress(0)
        stframe = st.empty()

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv5 model for frame detection
            results = model(frame)
            processed_frame = np.squeeze(results.render())
            
            out.write(processed_frame)

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels='RGB', use_container_width=True)

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

            # Maintain display frame rate
            time.sleep(1 / fps)

        cap.release()
        out.release()
        st.success("Video processing complete!")

# Handle video download
if download_button:
    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        with open(st.session_state.output_path, 'rb') as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
    else:
        st.sidebar.warning("No processed video available. Please process a video first.")
