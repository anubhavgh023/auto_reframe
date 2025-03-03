import cv2
import mediapipe as mp
import os
import subprocess

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Desired output dimensions (9:16 aspect ratio for Shorts)
output_width = 1080
output_height = 1920


def detect_faces(frame):
    """
    Detect faces in a given video frame using MediaPipe.
    Returns the bounding box of the detected face, or None if no face is found.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        # Return the largest face detected (assuming it's the main subject)
        largest_face = max(
            results.detections,
            key=lambda detection: (
                detection.location_data.relative_bounding_box.width
                * detection.location_data.relative_bounding_box.height
            ),
        )
        return largest_face.location_data.relative_bounding_box
    return None


def get_crop_box(bbox, frame_shape):
    """
    Calculate crop box dimensions centered on the face
    """
    original_width, original_height = frame_shape[1], frame_shape[0]
    target_aspect_ratio = output_width / output_height

    if bbox is None:
        # Default to center crop if no face detected
        crop_height = original_height
        crop_width = int(crop_height * target_aspect_ratio)
        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / target_aspect_ratio)
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)

    # Calculate face center
    x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
    y_center = int(bbox.ymin * original_height + (bbox.height * original_height) / 2)

    # Calculate crop dimensions
    crop_height = original_height
    crop_width = int(crop_height * target_aspect_ratio)

    # If crop width is too wide, adjust height
    if crop_width > original_width:
        crop_width = original_width
        crop_height = int(crop_width / target_aspect_ratio)

    # Calculate crop coordinates centered on face
    crop_x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
    crop_y1 = max(0, min(y_center - crop_height // 2, original_height - crop_height))

    return (crop_x1, crop_y1, crop_width, crop_height)


def reframe_video_with_audio(input_path, output_path):
    """
    Reframe the video to 9:16 aspect ratio with simple face tracking
    """
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    temp_output_path = "temp_output_video.mp4"
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
        crop_box = get_crop_box(bbox, frame.shape)

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
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete")

    # Release resources
    video_capture.release()
    output_video.release()

    print("Combining video with original audio...")
    # Use ffmpeg to combine video with original audio
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
            print("DONE !!")


if __name__ == "__main__":
    input_video_path = "../downloads/curated_videos/curated_video_01.mp4"  # Change this to your input video path
    output_video_path = "../downloads/output.mp4"  # Change this to your desired output path

    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
    else:
        reframe_video_with_audio(input_video_path, output_video_path)
