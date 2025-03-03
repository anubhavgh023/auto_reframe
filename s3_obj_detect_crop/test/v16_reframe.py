import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Desired output dimensions (9:16 aspect ratio for Shorts)
output_width = 1080
output_height = 1920


class StableFrameProcessor:
    def __init__(self, window_size=15, crop_margin=0.2):
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.size_history = deque(maxlen=window_size)
        self.last_valid_box = None
        self.crop_margin = crop_margin  # Margin around face for stability
        self.current_face_id = 0

    def get_stable_crop_box(self, bbox, frame_shape):
        original_height, original_width = frame_shape[:2]
        target_aspect = output_width / output_height

        if bbox is None:
            if self.last_valid_box is not None:
                return self.last_valid_box
            return self._get_center_crop(original_width, original_height)

        # Calculate face parameters with margin
        face_width = bbox.width * original_width * (1 + self.crop_margin)
        face_height = bbox.height * original_height * (1 + self.crop_margin)
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(
            bbox.ymin * original_height + (bbox.height * original_height) / 2
        )

        # Update history with weighted values
        self.x_history.append(x_center)
        self.y_history.append(y_center)
        self.size_history.append((face_width, face_height))

        # Calculate smoothed values
        smooth_x = int(np.mean(self.x_history))
        smooth_y = int(np.mean(self.y_history))
        avg_width, avg_height = np.mean(self.size_history, axis=0)

        # Calculate dynamic crop size based on face size
        crop_width = min(int(avg_width * 3), original_width)  # 3x face width
        crop_height = int(crop_width / target_aspect)

        if crop_height > original_height:
            crop_height = original_height
            crop_width = int(crop_height * target_aspect)

        # Ensure crop stays within frame bounds
        x1 = max(0, smooth_x - crop_width // 2)
        y1 = max(0, smooth_y - crop_height // 2)
        x2 = min(original_width, x1 + crop_width)
        y2 = min(original_height, y1 + crop_height)

        # Adjust if we hit frame boundaries
        if x2 - x1 < crop_width:
            x1 = max(0, x2 - crop_width)
        if y2 - y1 < crop_height:
            y1 = max(0, y2 - crop_height)

        crop_box = (x1, y1, x2 - x1, y2 - y1)
        self.last_valid_box = crop_box
        return crop_box

    def _get_center_crop(self, original_width, original_height):
        crop_height = min(
            original_height, int(original_width * output_height / output_width)
        )
        crop_width = int(crop_height * output_width / output_height)
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)


def adjust_brightness(frame):
    # Convert to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Apply CLAHE to Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

    # Convert back to BGR
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def detect_faces(frame):
    # Downscale for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        # Find largest face closest to center
        frame_center = (small_frame.shape[1] // 2, small_frame.shape[0] // 2)
        best_face = None
        max_score = -1

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            size = bbox.width * bbox.height
            x_center = bbox.xmin + bbox.width / 2
            y_center = bbox.ymin + bbox.height / 2
            distance = np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2)
            score = size * (1 - distance)

            if score > max_score:
                max_score = score
                best_face = bbox

        if best_face:
            # Scale back to original coordinates
            return type(
                "",
                (),
                {
                    "xmin": best_face.xmin * 2,
                    "ymin": best_face.ymin * 2,
                    "width": best_face.width * 2,
                    "height": best_face.height * 2,
                },
            )
    return None


def reframe_video_with_audio(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use temporary file for processing
    temp_output = "temp_reframed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, output_height))

    processor = StableFrameProcessor()
    frame_count = 0
    skip_frames = 1  # Process every other frame for detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Brightness adjustment
        frame = adjust_brightness(frame)

        # Face detection (every other frame)
        bbox = None
        if frame_count % (skip_frames + 1) == 0:
            bbox = detect_faces(frame)

        # Get crop coordinates
        crop_box = processor.get_stable_crop_box(bbox, frame.shape)
        x, y, w, h = crop_box

        # Crop and resize
        cropped = frame[y : y + h, x : x + w]
        resized = cv2.resize(
            cropped, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4
        )

        out.write(resized)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames:.1%})"
            )

    cap.release()
    out.release()

    # Combine with audio using faster encoding settings
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_output,
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-filter_complex",
        "[0:v][1:a]concat=n=1:v=1:a=1",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)


if __name__ == "__main__":
    input_path = "../downloads/curated_videos/curated_video_01.mp4"
    output_path = "../downloads/output.mp4"

    if os.path.exists(input_path):
        reframe_video_with_audio(input_path, output_path)
    else:
        print(f"Input file not found: {input_path}")
