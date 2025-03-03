import json
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Paths
input_video_dir = "../2.curation/assets/curated_videos"
output_video_dir = "../2.curation/assets/processed_shorts"
transitions_json = "transitions.json"

# Ensure output directory exists
os.makedirs(output_video_dir, exist_ok=True)

# Load OpenCV Face Detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_transitions(video_path, threshold=30.0):
    """
    Detects transition frames in a video based on frame difference.
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_idx = 0
    transitions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            score = np.mean(diff)

            if score > threshold:
                transitions.append(frame_idx)

        prev_frame = gray
        frame_idx += 1

    cap.release()
    return transitions


def generate_transitions_json():
    """
    Processes all videos in the input directory and detects transitions.
    Saves results to transitions.json.
    """
    transitions_data = {}

    for filename in os.listdir(input_video_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_video_dir, filename)
            print(f"Detecting transitions in {filename}...")
            transitions = detect_transitions(video_path)
            transitions_data[filename] = transitions

    with open(transitions_json, "w") as f:
        json.dump(transitions_data, f, indent=4)

    print(f"Transitions saved to {transitions_json}")


def find_face_position(frame):
    """
    Detects faces in the frame and returns the central y-coordinate for cropping.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) == 0:
        return None  # No face detected

    # Compute the center y-coordinate of the faces
    face_centers = [y + h // 2 for (x, y, w, h) in faces]
    return int(np.mean(face_centers))  # Return the average face center


def crop_to_9_16(video):
    """
    Crops a video to a 9:16 aspect ratio (1080x1920), ensuring faces remain in focus.
    """
    width, height = video.size
    target_width, target_height = 1080, 1920 # 9:16

    if width / height > target_width / target_height:
        # Crop width (landscape → vertical)
        new_width = int(height * (target_width / target_height))
        x1 = (width - new_width) // 2
        return video.crop(x1=x1, y1=0, x2=x1 + new_width, y2=height)
    else:
        # Crop height intelligently based on face position
        cap = cv2.VideoCapture(video.filename)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return video.crop(x1=0, y1=0, x2=width, y2=target_height)

        face_y = find_face_position(frame)
        if face_y is None:
            face_y = height // 2  # Default to center if no face detected

        new_height = int(width * (target_height / target_width))
        y1 = max(0, min(face_y - new_height // 2, height - new_height))

        return video.crop(x1=0, y1=y1, x2=width, y2=y1 + new_height)


def trim_and_crop_video(video_path, output_path, transition_frames, fps=30):
    """
    Trims a video by removing detected transition frames and crops it to 9:16.
    """
    video = VideoFileClip(video_path)
    final_clips = []
    start = 0

    for frame in transition_frames:
        end = frame / fps  # Convert frame index to seconds
        if end - start > 0.1:  # Ignore very short clips
            subclip = video.subclip(start, end)
            subclip = crop_to_9_16(subclip)  # Crop each clip
            final_clips.append(subclip)
        start = end + 0.1  # Skip a small buffer

    if start < video.duration:
        subclip = video.subclip(start, video.duration)
        subclip = crop_to_9_16(subclip)  # Crop final segment
        final_clips.append(subclip)

    if final_clips:
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(output_path, codec="libx264", fps=fps)


def process_videos():
    """
    Reads transitions.json, processes all videos by trimming and cropping.
    """
    with open(transitions_json, "r") as f:
        transitions_data = json.load(f)

    for filename, transition_frames in transitions_data.items():
        input_path = os.path.join(input_video_dir, filename)
        output_path = os.path.join(output_video_dir, filename)

        print(f"Processing {filename}...")
        trim_and_crop_video(input_path, output_path, transition_frames)
        print(f"Saved processed video: {output_path}")


if __name__ == "__main__":
    print("Generating transition data...")
    generate_transitions_json()

    print("\nProcessing videos...")
    process_videos()

    print("\nAll videos processed successfully!")
