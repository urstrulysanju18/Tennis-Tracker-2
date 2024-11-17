import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import os
import time

model_path = 'best.pt'  # Replace with your actual .pt file path

try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    # st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

# st.set_page_config(page_title="Tennis Game Tracking", layout="wide")

st.title("Tennis Game Tracking")

# Streamlit Sidebar for fancy, engaging user instructions
st.sidebar.title("Menu")

# # Main App UI
# st.title('🎾 Tennis Tracking App')
# st.write("Upload a tennis video to detect and track players in real-time.")

# # File uploader for video input
# uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# if uploaded_video is not None:
#     # Save the uploaded video to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
#         temp_video.write(uploaded_video.read())
#         temp_video_path = temp_video.name

#     # Open video capture
#     cap = cv2.VideoCapture(temp_video_path)
#     stframe = st.empty()

#     # Set up the output video
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
#         output_video_path = output_temp.name

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # Get total frames for progress bar
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     progress_bar = st.progress(0)
#     frame_count = 0

#     st.write("⏳ Processing video... Please wait.")

#     # Process video frames
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run detection model (mixed precision if CUDA is available)
#         if torch.cuda.is_available():
#             with torch.amp.autocast(device_type='cuda'):
#                 results = model(frame)
#         else:
#             results = model(frame)

#         frame = np.squeeze(results.render())  # Draw detection boxes on the frame

#         # Write frame to output video file
#         out.write(frame)

#         # Convert BGR to RGB for display
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame in Streamlit
#         stframe.image(frame, channels='RGB', use_container_width=True)

#         # Update progress bar
#         frame_count += 1
#         progress_bar.progress(frame_count / total_frames)

#         # Ensure consistent frame rate in display
#         time.sleep(1 / fps)

#     # Release video resources
#     cap.release()
#     out.release()

#     st.success("🎉 Video processing complete!")

#     # Provide download button for the processed video
#     st.write("📥 Download the processed video:")
#     with open(output_video_path, 'rb') as f:
#         st.download_button(
#             label="⬇ Download Processed Video",
#             data=f,
#             file_name="processed_video.mp4",
#             mime="video/mp4"
#         )

#     # Clean up temporary files
#     os.remove(temp_video_path)
#     os.remove(output_video_path)

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