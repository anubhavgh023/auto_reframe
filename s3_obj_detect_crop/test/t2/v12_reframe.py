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


class StabilizedPosition:
    def __init__(self, buffer_size=15):  # Increased buffer size for better stability
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.weights = np.linspace(0.5, 1.0, buffer_size)  # Progressive weights

    def update(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)

    def get_stable_position(self):
        if not self.x_buffer:
            return None

        # Apply weighted average with more weight to recent positions
        x = int(np.average(self.x_buffer, weights=self.weights[: len(self.x_buffer)]))
        y = int(np.average(self.y_buffer, weights=self.weights[: len(self.y_buffer)]))
        return (x, y)


class ColorEnhancer:
    @staticmethod
    def enhance_colors(frame):
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels
        lab = cv2.merge((l, a, b))

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Subtle saturation adjustment
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # Increase saturation by 20%
        s = np.clip(s, 0, 255)
        hsv = cv2.merge([h, s, v])

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class FaceTracker:
    def __init__(self):
        self.current_face_id = None
        self.face_positions = {}
        self.frames_without_detection = 0
        self.switch_threshold = 15  # Increased threshold for more stable switching
        self.last_crop_box = None
        self.crop_size = None
        self.stabilizer = StabilizedPosition(buffer_size=15)
        self.color_enhancer = ColorEnhancer()
        self.stable_crop_box = None
        self.transition_frames = 0
        self.max_transition_frames = 10

    def _calculate_face_signature(self, bbox, frame_shape):
        x_center = bbox.xmin + bbox.width / 2
        y_center = bbox.ymin + bbox.height / 2
        return (x_center, y_center)

    def _initialize_crop_size(self, frame_shape):
        original_width, original_height = frame_shape[1], frame_shape[0]
        target_aspect_ratio = output_width / output_height

        if original_width / original_height > target_aspect_ratio:
            crop_height = original_height
            crop_width = int(crop_height * target_aspect_ratio)
        else:
            crop_width = original_width
            crop_height = int(crop_width / target_aspect_ratio)

        self.crop_size = (crop_width, crop_height)

    def _get_crop_box(self, bbox, frame_shape):
        original_width, original_height = frame_shape[1], frame_shape[0]

        if self.crop_size is None:
            self._initialize_crop_size(frame_shape)

        crop_width, crop_height = self.crop_size

        # Calculate face center
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )

        # Update stabilizer
        self.stabilizer.update(x_center, y_center)
        stable_position = self.stabilizer.get_stable_position()

        if stable_position:
            x_center, y_center = stable_position

        # Calculate crop coordinates with bounds checking
        crop_x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
        crop_y1 = max(
            0, min(y_center - crop_height // 2, original_height - crop_height)
        )

        new_crop_box = (crop_x1, crop_y1, crop_width, crop_height)

        # Smooth transition between crop boxes
        if self.stable_crop_box is None:
            self.stable_crop_box = new_crop_box
        else:
            # Interpolate between old and new positions
            alpha = min(1.0, self.transition_frames / self.max_transition_frames)
            self.stable_crop_box = tuple(
                int(prev + (curr - prev) * alpha)
                for prev, curr in zip(self.stable_crop_box, new_crop_box)
            )
            self.transition_frames += 1

        return self.stable_crop_box

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

        # Check if this is a new face position with increased threshold
        new_face = True
        current_face_id = None
        for face_id, stored_sig in self.face_positions.items():
            if (
                abs(stored_sig[0] - face_sig[0]) < 0.15
                and abs(stored_sig[1] - face_sig[1]) < 0.15
            ):  # Increased threshold
                new_face = False
                current_face_id = face_id
                break

        if new_face:
            current_face_id = len(self.face_positions)
            self.face_positions[current_face_id] = face_sig

        # Handle speaker switching with reset
        if self.current_face_id != current_face_id:
            self.current_face_id = current_face_id
            self.stabilizer = StabilizedPosition(buffer_size=15)  # Reset stabilizer
            self.transition_frames = 0

        crop_box = self._get_crop_box(bbox, frame_shape)
        self.last_crop_box = crop_box
        return crop_box

    def _get_default_crop_box(self, frame_shape):
        if self.crop_size is None:
            self._initialize_crop_size(frame_shape)

        crop_width, crop_height = self.crop_size
        original_width, original_height = frame_shape[1], frame_shape[0]

        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)

    def process_frame(self, frame, crop_box):
        """Process frame with crop and color enhancement"""
        x1, y1, crop_w, crop_h = crop_box

        # Crop the frame
        cropped_frame = frame[int(y1) : int(y1 + crop_h), int(x1) : int(x1 + crop_w)]

        # Enhance colors
        enhanced_frame = self.color_enhancer.enhance_colors(cropped_frame)

        # Resize to final output dimensions
        return cv2.resize(
            enhanced_frame,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )


# Rest of the code (detect_faces and reframe_video_with_audio functions) remains the same,
# but use face_tracker.process_frame() instead of manual crop and resize
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
