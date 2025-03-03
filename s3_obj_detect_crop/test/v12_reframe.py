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

# Output dimensions
output_width = 1080
output_height = 1920
face_height = output_height // 2  # Each face gets half the height


class FixedFaceTracker:
    def __init__(self, target_person_index, window_size=30):
        self.target_person_index = (
            target_person_index  # 0 for first person, 1 for second
        )
        self.crop_box = None
        self.is_initialized = False

        # For initial position averaging
        self.initial_positions = []
        self.init_frames_needed = window_size

    def _calculate_initial_position(self, bbox, frame_shape):
        """Calculate and store initial crop position based on face detection"""
        original_width, original_height = frame_shape[1], frame_shape[0]

        # Calculate face center
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )
        face_width = int(
            bbox.width * original_width * 2
        )  # Double the detection width for better framing

        # Calculate crop dimensions for 1080x960 output (each face)
        crop_height = min(
            original_height, int(original_height * 0.8)
        )  # Use 80% of frame height
        crop_width = output_width * crop_height // face_height

        # Calculate crop coordinates
        crop_x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
        crop_y1 = max(
            0, min(y_center - crop_height // 2, original_height - crop_height)
        )

        return (crop_x1, crop_y1, crop_width, crop_height)

    def update(self, faces, frame_shape):
        """Update tracker with detected faces"""
        if faces[self.target_person_index] is None:
            return (
                self.crop_box
                if self.crop_box is not None
                else self._get_default_crop_box(frame_shape)
            )

        if not self.is_initialized:
            # Collecting initial positions
            position = self._calculate_initial_position(
                faces[self.target_person_index], frame_shape
            )
            self.initial_positions.append(position)

            if len(self.initial_positions) >= self.init_frames_needed:
                # Average the initial positions to get a stable crop box
                avg_position = [
                    int(
                        sum(pos[i] for pos in self.initial_positions)
                        / len(self.initial_positions)
                    )
                    for i in range(4)
                ]
                self.crop_box = tuple(avg_position)
                self.is_initialized = True
                print(
                    f"Fixed position initialized for person {self.target_person_index + 1}"
                )

            return self._get_default_crop_box(frame_shape)

        return self.crop_box

    def _get_default_crop_box(self, frame_shape):
        """Return center crop if no face is detected"""
        original_width, original_height = frame_shape[1], frame_shape[0]
        crop_height = min(original_height, int(original_height * 0.8))
        crop_width = output_width * crop_height // face_height
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)


def detect_faces(frame):
    """Detect faces and return bounding boxes for two largest faces"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if not results.detections:
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

    # Return bounding boxes for the two largest faces
    face1 = faces[0].location_data.relative_bounding_box if len(faces) > 0 else None
    face2 = faces[1].location_data.relative_bounding_box if len(faces) > 1 else None

    return face1, face2


def process_face_video(input_path, output_path, person_index):
    """Process video to track and extract one person's face with fixed camera"""
    video_capture = cv2.VideoCapture(input_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    face_tracker = FixedFaceTracker(person_index)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        output_path, fourcc, fps, (output_width, face_height)
    )

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect both faces
        face1_bbox, face2_bbox = detect_faces(frame)
        faces = [face1_bbox, face2_bbox]

        # Get fixed crop box for this person
        crop_box = face_tracker.update(faces, frame.shape)

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
            print(f"Processing person {person_index + 1}: {progress:.1f}% complete")

    video_capture.release()
    output_video.release()


def compose_final_video():
    """Stack the two face videos vertically and add original audio"""
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        temp_face1_path,
        "-i",
        temp_face2_path,
        "-i",
        input_video_path,
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
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

    try:
        # Process each person separately with fixed camera positions
        print("Processing first person (top video)...")
        process_face_video(input_video_path, temp_face1_path, 0)

        print("Processing second person (bottom video)...")
        process_face_video(input_video_path, temp_face2_path, 1)

        print("Composing final video...")
        compose_final_video()

        print(f"Successfully saved final video to: {output_video_path}")

    finally:
        # Cleanup temporary files
        for temp_file in [temp_face1_path, temp_face2_path]:
            if os.path.exists(temp_file):
                print("hi")


if __name__ == "__main__":
    main()
