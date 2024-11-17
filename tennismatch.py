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
model_path = 'best.pt'  # Replace with the path to your trained model
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

# Sidebar for user interaction
st.sidebar.title("Menu")
uploaded_video = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open the video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("⏳ Processing video... Please wait.")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 model inference on the frame
        results = model(frame)
        processed_frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write the processed frame to the output video file
        out.write(processed_frame)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels='RGB', use_container_width=True)

        # Update the progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate for display
        time.sleep(1 / fps)

    # Release resources
    cap.release()
    out.release()

    st.success("🎉 Video processing complete!")

    # Provide download button for the processed video
    st.write("📥 Download the processed video:")
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
else:
    st.sidebar.info("Upload a video file to get started.")
