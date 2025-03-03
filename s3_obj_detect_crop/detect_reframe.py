import json
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Paths
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
input_video_dir = os.path.join(script_dir, "../s2_curation/assets/curated_videos")
output_video_dir = os.path.join(script_dir, "../s2_curation/assets/processed_shorts")
transitions_json = os.path.join(script_dir, "transitions.json")

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

# ---
def find_face_positions(video_path, num_samples=10):
    """
    Samples multiple frames throughout the video and returns face positions.
    Returns a list of y-coordinates for detected faces.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames at regular intervals
    sample_indices = [i * total_frames // num_samples for i in range(num_samples)]
    face_positions = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        if len(faces) > 0:
            # Track face centers and heights for better positioning
            for (x, y, w, h) in faces:
                face_center = y + h // 2
                face_positions.append((face_center, h))

    cap.release()
    return face_positions, height

def crop_to_9_16(video):
    """
    Crops a video to a 9:16 aspect ratio (1080x1920), ensuring faces remain centered.
    """
    width, height = video.size
    target_width, target_height = 1080, 1920  # 9:16 aspect ratio

    if width / height > target_width / target_height:
        # Crop width (landscape → vertical)
        new_width = int(height * (target_width / target_height))
        # We can improve horizontal centering by face position too, but for now:
        x1 = (width - new_width) // 2
        return video.crop(x1=x1, y1=0, x2=x1 + new_width, y2=height)
    else:
        # Crop height intelligently based on face positions from multiple frames
        face_positions, frame_height = find_face_positions(video.filename)

        # Calculate the crop height based on target aspect ratio
        new_height = int(width * (target_height / target_width))

        # If we couldn't find any faces, default to center
        if not face_positions:
            y1 = max(0, min(height // 2 - new_height // 2, height - new_height))
            return video.crop(x1=0, y1=y1, x2=width, y2=y1 + new_height)

        # Calculate the ideal crop position based on detected faces
        # Prioritize larger/closer faces (those with larger height)
        # Sort by face height (descending)
        face_positions.sort(key=lambda x: x[1], reverse=True)

        # Take the top 3 largest faces (or fewer if less are available)
        top_faces = face_positions[:min(3, len(face_positions))]

        # Weight the positions by face size for a more balanced crop
        total_weight = sum(h for _, h in top_faces)
        weighted_center = sum(pos * h / total_weight for pos, h in top_faces)

        # Ensure the face is positioned in the upper third of the frame
        # (This is a common composition rule for portrait videos)
        target_position = new_height * 0.4  # 40% from top

        # Calculate y1 to position the face at the target position
        y1 = int(weighted_center - target_position)

        # Make sure y1 stays within bounds
        y1 = max(0, min(y1, height - new_height))

        return video.crop(x1=0, y1=y1, x2=width, y2=y1 + new_height)
# ----


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
