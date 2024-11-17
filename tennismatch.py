import streamlit as st
import cv2
import torch
import tempfile
from pathlib import Path
import numpy as np

# Fix path compatibility for Windows
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model_path = 'best.pt'
model = torch.hub.load('.', 'custom', path=model_path, source='local')

# Streamlit page configuration
st.set_page_config(page_title="Tennis Game Tracking", layout="wide")

# Title of the application
st.title("Tennis Game Tracking")

# Initialize flags and file paths
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'input_file_name' not in st.session_state:
    st.session_state.input_file_name = None
if 'show_input_video' not in st.session_state:
    st.session_state.show_input_video = False

# Sidebar for navigation
st.sidebar.title("Menu")
selected_option = st.sidebar.radio("Choose an option:", ["Upload and Preview Video", "Process Video", "Download Processed Video"])

# File uploader in the sidebar
input_file = st.sidebar.file_uploader("Select Input File", type=["mp4", "mov", "avi"])

# Set input file name if a file is uploaded
if input_file:
    st.session_state.input_file_name = Path(input_file.name).stem  # Extract the file name without extension

# Handle selected option
if selected_option == "Upload and Preview Video":
    if input_file:
        st.session_state.show_input_video = True
        st.subheader("Input Video Preview:")
        st.video(input_file)
    else:
        st.sidebar.warning("Please select a video file to preview.")

elif selected_option == "Process Video":
    if input_file:
        st.session_state.show_input_video = False  # Hide the input video during processing
        # Save the uploaded file temporarily
        temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_input_path, "wb") as f:
            f.write(input_file.read())

        # Open the input video
        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a temporary output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        st.session_state.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Image display placeholder for frame-by-frame processing
        st.subheader("Processing Video...")
        frame_display = st.empty()

        # Processing each frame
        frame_num = 0
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on the frame using YOLOv5
            results = model(frame)
            processed_frame = np.squeeze(results.render())  # Render the processed frame
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Show the processed frame in Streamlit
            frame_display.image(frame_rgb, channels='RGB', use_column_width=True)

            # Write the processed frame to output
            out.write(processed_frame)

            # Update progress bar
            frame_num += 1
            progress_percentage = int((frame_num / total_frames) * 100)
            progress_bar.progress(progress_percentage)

        cap.release()
        out.release()
        st.success("Video processing complete!")

        # Automatically show the processed video after processing
        if st.session_state.output_path and Path(st.session_state.output_path).exists():
            st.subheader("Processed Output Video:")
            st.video(st.session_state.output_path)

    else:
        st.sidebar.warning("Please select a video file to process.")

elif selected_option == "Download Processed Video":
    st.subheader("Download Processed Video:")
    if st.session_state.output_path and Path(st.session_state.output_path).exists():
        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=f"{st.session_state.input_file_name}_output.mp4",
                mime="video/mp4"
            )
    else:
        st.sidebar.warning("No processed video available. Please upload and process a video first.")