import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from collections import deque
import logging
import absl.logging
from deepface import DeepFace
import threading

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)


class PersonIdentifier:
    def __init__(self):
        self.person1_embedding = None
        self.person2_embedding = None
        self.is_initialized = False
        self.lock = threading.Lock()

    def initialize_from_frame(self, frame):
        """Initialize person embeddings from first clear frame with two faces"""
        try:
            # Detect and analyze faces in the frame
            results = DeepFace.extract_faces(frame, enforce_detection=True)

            if len(results) >= 2:
                # Get embeddings for the two largest faces
                embeddings = []
                for face_info in results[:2]:
                    face_img = face_info["face"]
                    embedding = DeepFace.represent(face_img, model_name="Facenet")
                    embeddings.append(embedding)

                with self.lock:
                    self.person1_embedding = embeddings[0]
                    self.person2_embedding = embeddings[1]
                    self.is_initialized = True
                return True
        except Exception as e:
            print(f"Failed to initialize face embeddings: {e}")
        return False

    def identify_face(self, face_img):
        """Identify if a face image belongs to person 1 or 2"""
        if not self.is_initialized:
            return None

        try:
            face_embedding = DeepFace.represent(face_img, model_name="Facenet")

            # Calculate similarity with both stored embeddings
            distance1 = self._calculate_distance(face_embedding, self.person1_embedding)
            distance2 = self._calculate_distance(face_embedding, self.person2_embedding)

            # Return the closest match if it's within threshold
            threshold = 0.6  # Adjust based on testing
            if min(distance1, distance2) > threshold:
                return None

            return 0 if distance1 < distance2 else 1

        except Exception:
            return None

    def _calculate_distance(self, embedding1, embedding2):
        """Calculate cosine distance between face embeddings"""
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))


class PersonTracker:
    def __init__(self, target_person_id, frame_shape):
        self.target_person_id = target_person_id
        self.crop_box = self._get_default_crop_box(frame_shape)
        self.frame_shape = frame_shape

    def _get_default_crop_box(self, frame_shape):
        original_width, original_height = frame_shape[1], frame_shape[0]
        crop_height = min(original_height, int(original_height * 0.8))
        crop_width = 1080 * crop_height // 960  # maintain aspect ratio
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)

    def update(self, frame, face_detections, person_identifier):
        """Update tracking for specific person"""
        if not face_detections:
            return self.crop_box

        original_width, original_height = self.frame_shape[1], self.frame_shape[0]

        for detection in face_detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * original_width)
            y1 = int(bbox.ymin * original_height)
            width = int(bbox.width * original_width)
            height = int(bbox.height * original_height)

            # Extract face image
            face_img = frame[y1 : y1 + height, x1 : x1 + width]
            if face_img.size == 0:
                continue

            # Identify the person
            person_id = person_identifier.identify_face(face_img)

            if person_id == self.target_person_id:
                # Calculate crop box for this person
                crop_height = min(original_height, int(height * 3))
                crop_width = 1080 * crop_height // 960

                crop_x1 = max(
                    0,
                    min(x1 + width // 2 - crop_width // 2, original_width - crop_width),
                )
                crop_y1 = max(
                    0,
                    min(
                        y1 + height // 2 - crop_height // 2,
                        original_height - crop_height,
                    ),
                )

                return (crop_x1, crop_y1, crop_width, crop_height)

        return self.crop_box


def process_video(input_path, output_path1, output_path2):
    """Process video to create separate tracks for each person"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out1 = cv2.VideoWriter(output_path1, fourcc, fps, (1080, 960))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (1080, 960))

    # Initialize person identifier and trackers
    person_identifier = PersonIdentifier()
    person1_tracker = PersonTracker(0, (frame_height, frame_width))
    person2_tracker = PersonTracker(1, (frame_height, frame_width))

    # First pass: Initialize person identifiers
    print("Initializing face recognition...")
    while not person_identifier.is_initialized and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if person_identifier.initialize_from_frame(frame):
            break

    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Second pass: Process video
    print("Processing video...")
    frame_count = 0
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.7
    ) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.detections:
                continue

            # Update trackers for each person
            crop_box1 = person1_tracker.update(
                frame, results.detections, person_identifier
            )
            crop_box2 = person2_tracker.update(
                frame, results.detections, person_identifier
            )

            # Process frames for each person
            for crop_box, out in [(crop_box1, out1), (crop_box2, out2)]:
                x1, y1, w, h = crop_box
                cropped = frame[int(y1) : int(y1 + h), int(x1) : int(x1 + w)]
                resized = cv2.resize(cropped, (1080, 960))
                out.write(resized)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")

    # Release resources
    cap.release()
    out1.release()
    out2.release()


def compose_final_video(input_path, face1_path, face2_path, output_path):
    """Compose final video with faces stacked vertically"""
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        face1_path,
        "-i",
        face2_path,
        "-i",
        input_path,
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
        output_path,
    ]
    subprocess.run(ffmpeg_command, check=True)


def main():
    input_path = "../downloads/curated_videos/curated_video_01.mp4"
    temp_face1_path = "../downloads/face_01.mp4"
    temp_face2_path = "../downloads/face_02.mp4"
    output_path = "../downloads/final_composition.mp4"

    try:
        process_video(input_path, temp_face1_path, temp_face2_path)
        compose_final_video(input_path, temp_face1_path, temp_face2_path, output_path)
        print(f"Successfully created video: {output_path}")
    finally:
        # Cleanup temporary files
        for temp_file in [temp_face1_path, temp_face2_path]:
            if os.path.exists(temp_file):
                print("Hi")


if __name__ == "__main__":
    main()