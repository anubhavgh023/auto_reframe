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


def trim_transitions(video_path, output_path, transition_frames, fps=30):
    """
    Trims a video by removing detected transition frames.
    """
    video = VideoFileClip(video_path)
    final_clips = []
    start = 0

    for frame in transition_frames:
        end = frame / fps  # Convert frame index to seconds
        if end - start > 0.1:  # Ignore very short clips
            final_clips.append(video.subclip(start, end))
        start = end + 0.1  # Skip a small buffer

    if start < video.duration:
        final_clips.append(video.subclip(start, video.duration))

    if final_clips:
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(output_path, codec="libx264", fps=fps)


def process_videos():
    """
    Reads transitions.json and processes all videos accordingly.
    """
    with open(transitions_json, "r") as f:
        transitions_data = json.load(f)

    for filename, transition_frames in transitions_data.items():
        input_path = os.path.join(input_video_dir, filename)
        output_path = os.path.join(output_video_dir, filename)

        print(f"Processing {filename}...")
        trim_transitions(input_path, output_path, transition_frames)
        print(f"Saved processed video: {output_path}")


if __name__ == "__main__":
    print("Generating transition data...")
    generate_transitions_json()

    print("\nProcessing videos...")
    process_videos()

    print("\nAll videos processed successfully!")
