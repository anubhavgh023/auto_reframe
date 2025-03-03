# fixed camera, issue is framing of face
# face going out of frame
# fix audio: no audio in shorts
import cv2
import numpy as np
import mediapipe as mp
import os


class PodcastToShortsConverter:
    def __init__(self, input_video_path, output_dir):
        """
        Initialize the converter with input video and output directory

        Args:
            input_video_path (str): Path to the input horizontal video
            output_dir (str): Directory to save processed vertical videos
        """
        self.input_video_path = input_video_path
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # MediaPipe face detection setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        # Video capture and properties
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 9:16 target dimensions
        self.target_width = 1080
        self.target_height = 1920

    def detect_faces(self, frame):
        """
        Detect faces in a frame using MediaPipe

        Args:
            frame (np.ndarray): Input video frame

        Returns:
            list: List of detected face locations
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        ) as face_detection:

            results = face_detection.process(rgb_frame)
            faces = []

            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame.shape[1])
                    y = int(bbox.ymin * frame.shape[0])
                    w = int(bbox.width * frame.shape[1])
                    h = int(bbox.height * frame.shape[0])

                    faces.append((x, y, w, h))

            return faces

    def crop_to_vertical(self, frame, faces):
        """
        Crop frame to capture all faces while maintaining original proportions

        Args:
            frame (np.ndarray): Input frame
            faces (list): List of face rectangles

        Returns:
            np.ndarray: Cropped vertical frame
        """
        # If no faces detected, return original frame
        if not faces:
            return cv2.resize(frame, (self.target_width, self.target_height))

        # Find the leftmost and rightmost points of all faces
        min_x = min(face[0] for face in faces)
        max_x = max(face[0] + face[2] for face in faces)

        # Target aspect ratio
        target_aspect = 9 / 16

        # Compute crop width to include all faces
        crop_width = max_x - min_x + int(0.2 * self.width)  # Add 20% padding
        crop_height = self.height

        # Center the crop around the detected faces
        center_x = (min_x + max_x) // 2
        start_x = max(0, center_x - crop_width // 2)

        # Ensure crop stays within frame
        if start_x + crop_width > self.width:
            start_x = self.width - crop_width

        # Crop
        cropped = frame[0:crop_height, start_x : start_x + crop_width]

        # Resize to target dimensions using INTER_LINEAR for better quality
        resized = cv2.resize(
            cropped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def process_video(self, min_face_duration=0.5):
        """
        Process video, detect faces, and create vertical shorts

        Args:
            min_face_duration (float): Minimum duration (seconds) to track faces

        Returns:
            str: Path to output video
        """
        # Prepare video writer
        output_filename = (
            os.path.basename(self.input_video_path).split(".")[0] + "_shorts.mp4"
        )
        output_path = os.path.join(self.output_dir, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, self.fps, (self.target_width, self.target_height)
        )

        # Face tracking variables
        current_faces = None
        face_start_time = 0
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            current_time = frame_count / self.fps

            if faces:
                if current_faces is None:
                    # New faces detected
                    current_faces = faces
                    face_start_time = current_time

                elif not self._are_faces_similar(current_faces, faces):
                    # Faces changed, check if previous faces were shown long enough
                    if current_time - face_start_time >= min_face_duration:
                        # Write previous face frames
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._write_face_segment(out, current_faces)

                    # Reset for new faces
                    current_faces = faces
                    face_start_time = current_time

            frame_count += 1

        # Close video objects
        self.cap.release()
        out.release()

        return output_path

    def _are_faces_similar(self, faces1, faces2, iou_threshold=0.3):
        """
        Check if two sets of faces are similar

        Args:
            faces1 (list): First set of faces
            faces2 (list): Second set of faces
            iou_threshold (float): Intersection over Union threshold

        Returns:
            bool: Whether face sets are considered similar
        """
        if len(faces1) != len(faces2):
            return False

        # Check pairwise IoU
        for f1 in faces1:
            best_iou = 0
            for f2 in faces2:
                iou = self._compute_iou(f1, f2)
                best_iou = max(best_iou, iou)

            if best_iou < iou_threshold:
                return False

        return True

    def _compute_iou(self, rect1, rect2):
        """
        Compute Intersection over Union for two rectangles

        Args:
            rect1 (tuple): First rectangle (x, y, w, h)
            rect2 (tuple): Second rectangle (x, y, w, h)

        Returns:
            float: Intersection over Union
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Compute intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # Compute intersection area
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        # Compute union area
        union_area = w1 * h1 + w2 * h2 - intersection_area

        # Compute IoU
        return intersection_area / union_area if union_area > 0 else 0

    def _write_face_segment(self, video_writer, faces):
        """
        Write a segment of video capturing all specified faces

        Args:
            video_writer (cv2.VideoWriter): Output video writer
            faces (list): List of face rectangles to capture
        """
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Crop and write vertical frame
            vertical_frame = self.crop_to_vertical(frame, faces)
            video_writer.write(vertical_frame)


def convert_podcast_to_shorts(input_dir, output_dir):
    """
    Convert all podcast videos in input directory to vertical shorts

    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save processed videos
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            converter = PodcastToShortsConverter(input_path, output_dir)
            output_path = converter.process_video()
            print(f"Processed {filename} → {output_path}")


# Example usage
input_video_dir = "../2.curation/assets/curated_videos"
output_video_dir = "../2.curation/assets/processed_shorts"
convert_podcast_to_shorts(input_video_dir, output_video_dir)
