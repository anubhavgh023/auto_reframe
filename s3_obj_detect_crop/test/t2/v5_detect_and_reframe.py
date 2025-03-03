import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Paths for input and output video
input_video_path = "../downloads/curated_video.mp4"
output_video_path = (
    "../downloads/reframed_video.mp4"  # Final output video with audio included
)

# Desired output dimensions (9:16 aspect ratio for Shorts)
output_width = 720
output_height = 1280 

# Smoothing parameters
smooth_factor = 0.85  # Higher value = smoother but slower response
prev_x_center, prev_y_center = None, None  # Track previous face positions


def detect_faces(frame):
    """
    Detect faces in a given video frame using MediaPipe.
    Returns the bounding box of the detected face, or None if no face is found.
    """
    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # Convert BGR to RGB for MediaPipe
    results = face_detection.process(rgb_frame)  # Perform face detection

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            return bbox  # Return the bounding box of the first detected face
    return None


def apply_smoothing(x_center, y_center):
    """
    Apply smoothing to the face position to stabilize the cropping.
    """
    global prev_x_center, prev_y_center
    if prev_x_center is not None and prev_y_center is not None:
        x_center = int(smooth_factor * prev_x_center + (1 - smooth_factor) * x_center)
        y_center = int(smooth_factor * prev_y_center + (1 - smooth_factor) * y_center)
    prev_x_center, prev_y_center = x_center, y_center
    return x_center, y_center


def reframe_video_with_audio(input_path, output_path):
    """
    Reframe the video to 9:16 aspect ratio by detecting faces and cropping the video accordingly.
    Apply smoothing to avoid abrupt movements and ensure audio sync.
    """
    # Open the input video
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get the original video properties
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Initialize temporary video writer without audio
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output_path = "../downloads/temp_output_video.mp4"
    output_video = cv2.VideoWriter(
        temp_output_path, fourcc, fps, (output_width, output_height)
    )

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit loop when the video ends

        # Detect face and get bounding box
        bbox = detect_faces(frame)
        if bbox:
            # Calculate the bounding box in pixel coordinates
            x1 = int(bbox.xmin * original_width)
            y1 = int(bbox.ymin * original_height)
            width = int(bbox.width * original_width)
            height = int(bbox.height * original_height)

            # Get the center of the bounding box
            x_center = x1 + width // 2
            y_center = y1 + height // 2

            # Apply smoothing to the center position
            x_center, y_center = apply_smoothing(x_center, y_center)

            # Calculate the cropping box for 9:16 aspect ratio
            crop_x1 = max(0, x_center - output_width // 2)
            crop_y1 = max(0, y_center - output_height // 2)
            crop_x2 = min(original_width, crop_x1 + output_width)
            crop_y2 = min(original_height, crop_y1 + output_height)

            # Crop and resize the frame to the desired output size
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height))

            # Write the processed frame to the temporary output video
            output_video.write(resized_frame)

    # Release the video objects
    video_capture.release()
    output_video.release()

    # Use ffmpeg to combine the processed video with the original audio
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        temp_output_path,  # Input processed video
        "-i",
        input_path,  # Input original video for audio
        "-c:v",
        "copy",  # Copy the video stream
        "-c:a",
        "aac",  # Set audio codec
        "-map",
        "0:v:0",  # Use video stream from processed video
        "-map",
        "1:a:0",  # Use audio stream from original video
        "-shortest",  # Ensure the output matches the shortest input (video or audio)
        output_path,  # Output file
    ]

    # Execute the ffmpeg command
    subprocess.run(ffmpeg_command, check=True)

    # Optionally, remove the temporary output video
    os.remove(temp_output_path)
    print(f"Reframed video with audio saved to: {output_path}")


if __name__ == "__main__":
    # Check if the input video file exists
    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
    else:
        # Process the video by detecting faces and cropping directly with audio
        reframe_video_with_audio(input_video_path, output_video_path)