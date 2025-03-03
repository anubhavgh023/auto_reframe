import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Paths for input and output video
input_video_path = "../downloads/curated_videos/curated_video_01.mp4"
output_video_path = "../downloads/reframed_video_01.mp4"

# Desired output dimensions (9:16 aspect ratio for Shorts)
output_width = 720
output_height = 1280

# Enhanced smoothing parameters
window_size = 30  # Number of frames for rolling average
position_history = deque(maxlen=window_size)
smooth_factor = 0.85


class FaceTracker:
    def __init__(self, window_size=30):
        self.position_history = deque(maxlen=window_size)
        self.prev_crop_box = None

    def update(self, bbox, frame_shape):
        if bbox is None:
            return (
                self.prev_crop_box
                if self.prev_crop_box is not None
                else self._get_default_crop_box(frame_shape)
            )

        original_width, original_height = frame_shape[1], frame_shape[0]

        # Calculate face center
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )

        # Add to position history
        self.position_history.append((x_center, y_center))

        # Calculate smoothed center using rolling average
        if len(self.position_history) > 0:
            smoothed_x = int(np.mean([p[0] for p in self.position_history]))
            smoothed_y = int(np.mean([p[1] for p in self.position_history]))
        else:
            smoothed_x, smoothed_y = x_center, y_center

        # Calculate crop box with padding
        face_width = bbox.width * original_width
        face_height = bbox.height * original_height
        padding_factor = 2.0  # Adjust this to control how much space around the face

        crop_width = max(output_width, int(face_width * padding_factor))
        crop_height = int(crop_width * (output_height / output_width))

        # Calculate crop coordinates
        crop_x1 = smoothed_x - crop_width // 2
        crop_y1 = smoothed_y - crop_height // 2

        # Ensure crop box stays within frame bounds
        crop_x1 = max(0, min(crop_x1, original_width - crop_width))
        crop_y1 = max(0, min(crop_y1, original_height - crop_height))

        crop_box = (crop_x1, crop_y1, crop_width, crop_height)

        # Smooth transition between crop boxes
        if self.prev_crop_box is not None:
            crop_box = self._smooth_crop_box(self.prev_crop_box, crop_box)

        self.prev_crop_box = crop_box
        return crop_box

    def _smooth_crop_box(self, prev_box, current_box):
        return tuple(
            int(prev * smooth_factor + current * (1 - smooth_factor))
            for prev, current in zip(prev_box, current_box)
        )

    def _get_default_crop_box(self, frame_shape):
        original_width, original_height = frame_shape[1], frame_shape[0]
        x1 = (original_width - output_width) // 2
        y1 = (original_height - output_height) // 2
        return (x1, y1, output_width, output_height)


def detect_faces(frame):
    """
    Detect faces in a given video frame using MediaPipe.
    Returns the bounding box of the detected face, or None if no face is found.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        # Return the largest face detected (assuming it's the main subject)
        largest_face = max(
            results.detections,
            key=lambda detection: (
                detection.location_data.relative_bounding_box.width
                * detection.location_data.relative_bounding_box.height
            ),
        )
        return largest_face.location_data.relative_bounding_box
    return None


def apply_high_quality_resize(frame, target_width, target_height):
    """
    Apply high-quality resize with Lanczos interpolation and sharpening.
    """
    # Resize with Lanczos interpolation
    resized = cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Apply subtle sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
    sharpened = cv2.filter2D(resized, -1, kernel)

    # Blend the sharpened image with the original
    result = cv2.addWeighted(resized, 0.7, sharpened, 0.3, 0)

    return result


def reframe_video_with_audio(input_path, output_path):
    """
    Reframe the video to 9:16 aspect ratio with improved quality and smoother tracking.
    """
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize face tracker
    face_tracker = FaceTracker(window_size)

    # Initialize video writer with high quality settings
    temp_output_path = "../downloads/temp_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        temp_output_path, fourcc, fps, (output_width, output_height)
    )

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect face and get crop box
        bbox = detect_faces(frame)
        crop_box = face_tracker.update(bbox, frame.shape)

        # Extract crop coordinates
        x1, y1, crop_w, crop_h = crop_box

        # Ensure we don't exceed frame boundaries
        x2 = min(x1 + crop_w, original_width)
        y2 = min(y1 + crop_h, original_height)

        # Crop and resize with high quality
        cropped_frame = frame[int(y1) : int(y2), int(x1) : int(x2)]
        processed_frame = apply_high_quality_resize(
            cropped_frame, output_width, output_height
        )

        # Write the frame
        output_video.write(processed_frame)

        # Progress indication
        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete")

    # Release resources
    video_capture.release()
    output_video.release()

    print("Combining video with original audio...")
    # Use ffmpeg with high quality settings
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        temp_output_path,
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "slow",  # Higher quality encoding
        "-crf",
        "18",  # High quality (lower = better, 18-28 is good range)
        "-c:a",
        "aac",
        "-b:a",
        "192k",  # High quality audio
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Successfully saved reframed video to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg processing: {e}")
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


if __name__ == "__main__":
    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
    else:
        reframe_video_with_audio(input_video_path, output_video_path)