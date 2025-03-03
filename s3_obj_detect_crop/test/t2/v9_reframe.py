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

# Reduced zoom factor for wider framing
zoom_factor = 0.8  # Smaller number means wider shot


class FaceTracker:
    def __init__(self):
        self.current_face_id = None
        self.face_positions = {}  # Store fixed positions for each detected face
        self.frames_without_detection = 0
        self.switch_threshold = 10  # Number of frames before considering a switch
        self.last_crop_box = None

    def _calculate_face_signature(self, bbox, frame_shape):
        """Calculate a signature for face position to identify different speakers"""
        x_center = bbox.xmin + bbox.width / 2
        y_center = bbox.ymin + bbox.height / 2
        return (x_center, y_center)

    def _get_crop_box(self, bbox, frame_shape):
        """Calculate crop box with wider framing"""
        original_width, original_height = frame_shape[1], frame_shape[0]
        target_aspect_ratio = output_width / output_height

        # Calculate face center
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )

        # Calculate crop dimensions with wider framing
        face_height = int(bbox.height * original_height / zoom_factor)
        crop_height = min(face_height * 2.5, original_height)  # Wider vertical framing
        crop_width = int(crop_height * target_aspect_ratio)

        # Adjust if crop is too wide
        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / target_aspect_ratio)

        # Calculate crop coordinates
        crop_x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
        crop_y1 = max(
            0, min(y_center - crop_height // 2, original_height - crop_height)
        )

        return (crop_x1, crop_y1, crop_width, crop_height)

    def update(self, bbox, frame_shape):
        if bbox is None:
            self.frames_without_detection += 1
            # Return last known crop box if available
            return (
                self.last_crop_box
                if self.last_crop_box is not None
                else self._get_default_crop_box(frame_shape)
            )

        # Reset counter when face is detected
        self.frames_without_detection = 0

        # Calculate face signature
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

        # If new face position found, add it
        if new_face:
            current_face_id = len(self.face_positions)
            self.face_positions[current_face_id] = face_sig

        # Switch focus if face ID changed
        if self.current_face_id != current_face_id:
            self.current_face_id = current_face_id

        # Calculate new crop box
        crop_box = self._get_crop_box(bbox, frame_shape)
        self.last_crop_box = crop_box
        return crop_box

    def _get_default_crop_box(self, frame_shape):
        """Default center crop with wider framing"""
        original_width, original_height = frame_shape[1], frame_shape[0]
        target_aspect_ratio = output_width / output_height

        crop_height = original_height
        crop_width = int(crop_height * target_aspect_ratio)

        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / target_aspect_ratio)

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
