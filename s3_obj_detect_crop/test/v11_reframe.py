import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Paths for input and output
input_video_path = "../downloads/curated_videos/curated_video_01.mp4"
temp_face1_path = "../downloads/face_01.mp4"
temp_face2_path = "../downloads/face_02.mp4"
output_video_path = "../downloads/final_composition.mp4"

# Output dimensions (9:16 aspect ratio for Shorts)
output_width = 1080
output_height = 1920
face_height = output_height // 2  # Each face gets half the height

# Enhanced stabilization parameters
window_size = 45
smooth_factor = 0.92
interpolation_factor = 0.1


class MultiFaceTracker:
    def __init__(self, face_index, window_size=45):
        self.face_index = face_index
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.width_history = deque(maxlen=window_size)
        self.height_history = deque(maxlen=window_size)
        self.prev_crop_box = None

    def _weighted_moving_average(self, history):
        if not history:
            return None
        weights = np.linspace(0.1, 1.0, len(history))
        weighted_values = np.array(history) * weights
        return np.sum(weighted_values) / np.sum(weights)

    def update(self, bbox, frame_shape):
        if bbox is None:
            return (
                self.prev_crop_box
                if self.prev_crop_box is not None
                else self._get_default_crop_box(frame_shape)
            )

        original_width, original_height = frame_shape[1], frame_shape[0]

        # Calculate face center and dimensions
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )
        face_width = int(bbox.width * original_width)
        face_height = int(bbox.height * original_height)

        # Update histories
        self.x_history.append(x_center)
        self.y_history.append(y_center)
        self.width_history.append(face_width)
        self.height_history.append(face_height)

        # Calculate stabilized values
        stabilized_x = int(self._weighted_moving_average(self.x_history))
        stabilized_y = int(self._weighted_moving_average(self.y_history))

        # Calculate crop dimensions for 1080x960 output (each face)
        crop_height = min(
            original_height, face_height * 3
        )  # Give some padding around the face
        crop_width = int(
            crop_height * (output_width / (output_height / 2))
        )  # Maintain aspect ratio

        # Calculate crop coordinates
        crop_x1 = max(
            0, min(stabilized_x - crop_width // 2, original_width - crop_width)
        )
        crop_y1 = max(
            0, min(stabilized_y - crop_height // 2, original_height - crop_height)
        )

        crop_box = (crop_x1, crop_y1, crop_width, crop_height)

        # Smooth transition
        if self.prev_crop_box is not None:
            crop_box = self._smooth_crop_box(self.prev_crop_box, crop_box)

        self.prev_crop_box = crop_box
        return crop_box

    def _smooth_crop_box(self, prev_box, current_box):
        smoothed_box = tuple(
            int(prev * smooth_factor + current * (1 - smooth_factor))
            for prev, current in zip(prev_box, current_box)
        )
        return tuple(
            int(smoothed + (current - smoothed) * interpolation_factor)
            for smoothed, current in zip(smoothed_box, current_box)
        )

    def _get_default_crop_box(self, frame_shape):
        original_width, original_height = frame_shape[1], frame_shape[0]
        crop_height = min(original_height, original_height // 2)
        crop_width = int(crop_height * (output_width / (output_height / 2)))
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)


def detect_faces(frame):
    """
    Detect faces in frame and return the two largest faces
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if not results.detections or len(results.detections) < 2:
        return None, None

    # Sort faces by size (largest to smallest)
    faces = sorted(
        results.detections,
        key=lambda detection: (
            detection.location_data.relative_bounding_box.width
            * detection.location_data.relative_bounding_box.height
        ),
        reverse=True,
    )

    return (
        faces[0].location_data.relative_bounding_box,
        faces[1].location_data.relative_bounding_box,
    )


def process_face_video(input_path, output_path, face_index):
    """
    Process video to track and extract one face
    """
    video_capture = cv2.VideoCapture(input_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    face_tracker = MultiFaceTracker(face_index)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        output_path, fourcc, fps, (output_width, face_height)
    )

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        face1_bbox, face2_bbox = detect_faces(frame)
        bbox = face1_bbox if face_index == 0 else face2_bbox

        # Get crop box for this face
        crop_box = face_tracker.update(bbox, frame.shape)

        if crop_box:
            x1, y1, crop_w, crop_h = crop_box
            cropped_frame = frame[
                int(y1) : int(y1 + crop_h), int(x1) : int(x1 + crop_w)
            ]
            processed_frame = cv2.resize(cropped_frame, (output_width, face_height))
            output_video.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing face {face_index + 1}: {progress:.1f}% complete")

    video_capture.release()
    output_video.release()


def compose_final_video():
    """
    Stack the two face videos vertically and add original audio
    """
    ffmpeg_command = [
        "ffmpeg",
        # Input face videos
        "-i",
        temp_face1_path,
        "-i",
        temp_face2_path,
        "-i",
        input_video_path,  # Original video for audio
        # Filter to stack videos vertically
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[v]",
        # Map stacked video and original audio
        "-map",
        "[v]",
        "-map",
        "2:a",
        # High quality encoding settings
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_video_path,
    ]

    subprocess.run(ffmpeg_command, check=True)


def main():
    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
        return

        # Process each face separately
        print("Processing first face...")
        process_face_video(input_video_path, temp_face1_path, 0)

        print("Processing second face...")
        process_face_video(input_video_path, temp_face2_path, 1)

        print("Composing final video...")
        compose_final_video()

        print(f"Successfully saved final video to: {output_video_path}")

    # finally:
    #     # Cleanup temporary files
    #     for temp_file in [temp_face1_path, temp_face2_path]:
    #         if os.path.exists(temp_file):
    #             os.remove(temp_file)


if __name__ == "__main__":
    main()
