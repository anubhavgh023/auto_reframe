import cv2
import mediapipe as mp
import os
from collections import deque
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Paths
INPUT_VIDEO = "../downloads/curated_videos/curated_video_01.mp4"
OUTPUT_DIR = "../downloads/temp_framed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video dimensions (9:16 aspect ratio)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


class FaceTracker:
    def __init__(self, window_size=30):
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.w_history = deque(maxlen=window_size)
        self.h_history = deque(maxlen=window_size)
        self.prev_box = None

    def update(self, bbox, frame_shape):
        if bbox is None:
            return self.prev_box if self.prev_box is not None else None

        frame_height, frame_width = frame_shape[:2]

        # Convert relative coordinates to absolute
        x = int(bbox.xmin * frame_width)
        y = int(bbox.ymin * frame_height)
        w = int(bbox.width * frame_width)
        h = int(bbox.height * frame_height)

        # Add to history
        self.x_history.append(x)
        self.y_history.append(y)
        self.w_history.append(w)
        self.h_history.append(h)

        # Calculate smooth coordinates
        smooth_x = int(np.mean(self.x_history))
        smooth_y = int(np.mean(self.y_history))
        smooth_w = int(np.mean(self.w_history))
        smooth_h = int(np.mean(self.h_history))

        # Expand box slightly
        expansion_factor = 1.5
        center_x = smooth_x + smooth_w // 2
        center_y = smooth_y + smooth_h // 2

        new_w = int(smooth_w * expansion_factor)
        new_h = int(smooth_h * expansion_factor)

        # Maintain aspect ratio
        if new_w / new_h > OUTPUT_WIDTH / OUTPUT_HEIGHT:
            new_h = int(new_w * (OUTPUT_HEIGHT / OUTPUT_WIDTH))
        else:
            new_w = int(new_h * (OUTPUT_WIDTH / OUTPUT_HEIGHT))

        # Calculate new box coordinates
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(frame_width, x1 + new_w)
        y1 = min(frame_height - new_h, y1)  # Ensure we don't go out of bounds

        box = (x1, y1, new_w, new_h)
        self.prev_box = box
        return box


def extract_faces(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize face trackers dict
    face_trackers = {}
    face_writers = {}

    print("Starting face extraction...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            # Process each detected face
            for i, detection in enumerate(results.detections):
                if i not in face_trackers:
                    print(f"New face detected! Creating tracker for face {i+1}")
                    face_trackers[i] = FaceTracker()
                    # Create video writer for this face
                    output_path = os.path.join(OUTPUT_DIR, f"face_{i+1:02d}.mp4")
                    face_writers[i] = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    )

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                crop_box = face_trackers[i].update(bbox, frame.shape)

                if crop_box:
                    x, y, w, h = crop_box
                    # Ensure we don't exceed frame boundaries
                    y2 = min(y + h, frame.shape[0])
                    x2 = min(x + w, frame.shape[1])
                    # Crop and resize
                    face_frame = frame[int(y) : int(y2), int(x) : int(x2)]
                    if face_frame.size > 0:  # Check if crop is valid
                        resized_face = cv2.resize(
                            face_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
                        )
                        face_writers[i].write(resized_face)

        # Progress update
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete")

    # Clean up
    cap.release()
    for writer in face_writers.values():
        writer.release()

    print(f"Face extraction complete! Found {len(face_trackers)} faces.")
    print(f"Output files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_faces(INPUT_VIDEO)
