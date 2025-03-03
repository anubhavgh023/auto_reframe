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
output_width = 1080
output_height = 1920


class FaceTracker:
    def __init__(self):
        self.current_face_id = None
        self.face_positions = {}  # Store fixed positions for each detected face
        self.frames_without_detection = 0
        self.switch_threshold = 10
        self.last_crop_box = None
        self.position_buffer = deque(maxlen=5)  # Small buffer for position smoothing
        self.crop_size = None  # Will be set on first frame

    def _calculate_face_signature(self, bbox, frame_shape):
        """Calculate a signature for face position to identify different speakers"""
        x_center = bbox.xmin + bbox.width / 2
        y_center = bbox.ymin + bbox.height / 2
        return (x_center, y_center)

    def _initialize_crop_size(self, frame_shape):
        """Initialize fixed crop size based on frame dimensions"""
        original_width, original_height = frame_shape[1], frame_shape[0]
        target_aspect_ratio = output_width / output_height

        # Calculate crop dimensions to maintain 9:16 ratio
        if original_width / original_height > target_aspect_ratio:
            # Width is limiting factor
            crop_height = original_height
            crop_width = int(crop_height * target_aspect_ratio)
        else:
            # Height is limiting factor
            crop_width = original_width
            crop_height = int(crop_width / target_aspect_ratio)

        self.crop_size = (crop_width, crop_height)

    def _get_crop_box(self, bbox, frame_shape):
        """Calculate crop box with fixed dimensions"""
        original_width, original_height = frame_shape[1], frame_shape[0]

        # Initialize crop size if not set
        if self.crop_size is None:
            self._initialize_crop_size(frame_shape)

        crop_width, crop_height = self.crop_size

        # Calculate face center
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )

        # Add position to buffer for smoothing
        self.position_buffer.append((x_center, y_center))

        # Calculate smoothed position
        if len(self.position_buffer) > 0:
            x_center = int(
                sum(p[0] for p in self.position_buffer) / len(self.position_buffer)
            )
            y_center = int(
                sum(p[1] for p in self.position_buffer) / len(self.position_buffer)
            )

        # Calculate crop coordinates
        crop_x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
        crop_y1 = max(
            0, min(y_center - crop_height // 2, original_height - crop_height)
        )

        return (crop_x1, crop_y1, crop_width, crop_height)

    def update(self, bbox, frame_shape):
        if bbox is None:
            self.frames_without_detection += 1
            return (
                self.last_crop_box
                if self.last_crop_box is not None
                else self._get_default_crop_box(frame_shape)
            )

        self.frames_without_detection = 0
        face_sig = self._calculate_face_signature(bbox, frame_shape)

        # Check if this is a new face position
        new_face = True
        current_face_id = None
        for face_id, stored_sig in self.face_positions.items():
            if (
                abs(stored_sig[0] - face_sig[0]) < 0.1
                and abs(stored_sig[1] - face_sig[1]) < 0.1
            ):
                new_face = False
                current_face_id = face_id
                break

        if new_face:
            current_face_id = len(self.face_positions)
            self.face_positions[current_face_id] = face_sig

        # Switch face ID and clear position buffer
        if self.current_face_id != current_face_id:
            self.current_face_id = current_face_id
            self.position_buffer.clear()

        crop_box = self._get_crop_box(bbox, frame_shape)
        self.last_crop_box = crop_box
        return crop_box

    def _get_default_crop_box(self, frame_shape):
        """Default center crop with fixed dimensions"""
        if self.crop_size is None:
            self._initialize_crop_size(frame_shape)

        crop_width, crop_height = self.crop_size
        original_width, original_height = frame_shape[1], frame_shape[0]

        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)


def detect_faces(frame):
    """
    Detect faces in a given video frame using MediaPipe.
    Returns the bounding box of the most prominent detected face.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        # Return the largest face detected (assuming it's the main speaker)
        largest_face = max(
            results.detections,
            key=lambda detection: (
                detection.location_data.relative_bounding_box.width
                * detection.location_data.relative_bounding_box.height
            ),
        )
        return largest_face.location_data.relative_bounding_box
    return None


def reframe_video_with_audio(input_path, output_path):
    """
    Reframe the video to 9:16 aspect ratio with direct cuts between speakers.
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
    face_tracker = FaceTracker()

    # Initialize video writer
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

        # Crop the frame
        cropped_frame = frame[int(y1) : int(y1 + crop_h), int(x1) : int(x1 + crop_w)]

        # Resize to final output dimensions
        processed_frame = cv2.resize(
            cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR
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
