import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Load the YOLO model
model_path = 'best.pt'  # Replace with your actual .pt file path

try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

# Configure Streamlit page
st.set_page_config(page_title="Tennis Game Tracking", layout="wide")

st.title("🎾 Tennis Game Tracking")

# Sidebar menu for user options
st.sidebar.title("Menu")
option = st.sidebar.radio(
    "Choose an action:",
    ("Upload Video", "Preview Video", "Process Video", "Download Video")
)

# Video file uploader
uploaded_video = None
if option == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Display preview if the user selects "Preview Video"
    if option == "Preview Video":
        st.video(temp_video_path)

    # Process video if the user selects "Process Video"
    if option == "Process Video":
        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()

        # Set up output video
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

            # Run detection model
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    results = model(frame)
            else:
                results = model(frame)

            frame = np.squeeze(results.render())  # Draw detection boxes on the frame
            out.write(frame)  # Write frame to output video

            # Convert BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels='RGB', use_container_width=True)

            # Update progress bar
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

            # Ensure consistent frame rate in display
            time.sleep(1 / fps)

        # Release video resources
        cap.release()
        out.release()

        st.success("🎉 Video processing complete!")

    # Display download button if the user selects "Download Video"
    if option == "Download Video":
        if 'output_video_path' in locals():
            st.write("📥 Download the processed video:")
            with open(output_video_path, 'rb') as f:
                st.sidebar.download_button(
                    label="⬇ Download Processed Video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

        else:
            st.sidebar.warning("Please process a video before downloading.")

    # Clean up temporary files after processing
    if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if 'output_video_path' in locals() and os.path.exists(output_video_path):
        os.remove(output_video_path)
