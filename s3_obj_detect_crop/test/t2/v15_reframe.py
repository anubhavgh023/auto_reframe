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
    def __init__(self, buffer_size=10):  # Reduced buffer for faster response
        self.positions = deque(maxlen=buffer_size)

    def update(self, x, y, w, h):
        self.positions.append((x, y, w, h))

    def get_stable_position(self):
        if not self.positions:
            return None

        # Use a moving average for stability
        x = int(np.mean([pos[0] for pos in self.positions]))
        y = int(np.mean([pos[1] for pos in self.positions]))
        w = int(np.mean([pos[2] for pos in self.positions]))
        h = int(np.mean([pos[3] for pos in self.positions]))

        return (x, y, w, h)


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

    # Subtle saturation boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # Increase saturation by 20%
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        return results.detections  # Return all detected faces
    return []


def reframe_video_with_audio(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    temp_output_path = "../downloads/temp_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        temp_output_path, fourcc, fps, (output_width, output_height)
    )

    # Calculate initial crop dimensions
    target_ratio = output_width / output_height
    if original_width / original_height > target_ratio:
        crop_height = original_height
        crop_width = int(crop_height * target_ratio)
    else:
        crop_width = original_width
        crop_height = int(crop_width / target_ratio)

    # Store the bounding box of the "active" speaker
    active_speaker_box = None
    frame_count = 0
    stabilizers = {}  # Dictionary to hold a StabilizedPosition object for each face

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        detections = detect_faces(frame)

        if detections:
            # Determine the "active" speaker (you'll need a way to determine this).
            # This is a placeholder - replace with your logic.
            # For example, you might use audio analysis or a manual override.
            # This example just picks the first detected face each time.
            active_speaker = detections[0]

            # Get the ID of the active speaker (using the index in the detections list)
            active_speaker_id = detections.index(active_speaker)

            # If there isn't a stabilizer for this face, create one
            if active_speaker_id not in stabilizers:
                stabilizers[active_speaker_id] = StabilizedPosition()

            # Convert relative coordinates to absolute
            x = int(
                active_speaker.location_data.relative_bounding_box.xmin * original_width
            )
            y = int(
                active_speaker.location_data.relative_bounding_box.ymin
                * original_height
            )
            w = int(
                active_speaker.location_data.relative_bounding_box.width
                * original_width
            )
            h = int(
                active_speaker.location_data.relative_bounding_box.height
                * original_height
            )

            # Update the stabilizer for the active speaker
            stabilizers[active_speaker_id].update(x, y, w, h)
            stable_pos = stabilizers[active_speaker_id].get_stable_position()
            if stable_pos:
                x, y, w, h = stable_pos
                active_speaker_box = (x, y, w, h)
            else:
                active_speaker_box = (
                    x,
                    y,
                    w,
                    h,
                )  # Use the unstabilized position temporarily

        if active_speaker_box:
            x, y, w, h = active_speaker_box

            # Calculate crop box centered on the active speaker
            crop_x = max(
                0, min(x + w // 2 - crop_width // 2, original_width - crop_width)
            )
            crop_y = max(
                0, min(y + h // 2 - crop_height // 2, original_height - crop_height)
            )

        else:
            # Default center crop if no face is detected
            crop_x = (original_width - crop_width) // 2
            crop_y = (original_height - crop_height) // 2

        # Crop frame
        cropped_frame = frame[
            crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
        ]

        # Enhance colors
        enhanced_frame = enhance_colors(cropped_frame)

        # Resize to output dimensions
        final_frame = cv2.resize(
            enhanced_frame,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Write frame
        output_video.write(final_frame)

        # Progress indication
        frame_count += 1
        if frame_count % 30 == 0:
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
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
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