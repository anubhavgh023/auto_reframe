# import subprocess


# def add_background_music(video_path, bg_music_path, output_path, bg_volume=0.3):
#     """
#     Adds background music to a video using ffmpeg and lowers the volume of the background music.

#     :param video_path: Path to the input video file.
#     :param bg_music_path: Path to the background music file.
#     :param output_path: Path to save the output video file.
#     :param bg_volume: Volume level for the background music (default is 0.3, which is 30% of the original volume).
#     """
#     # FFmpeg command to add background music to the video with adjusted volume
#     command = [
#         "ffmpeg",
#         "-i",
#         video_path,  # Input video file
#         "-i",
#         bg_music_path,  # Input background music file
#         "-filter_complex",  # Use filter complex for audio mixing
#         f"[1:a]volume={bg_volume}[bg];[0:a][bg]amix=inputs=2:duration=shortest",  # Lower volume of bg music and mix
#         "-c:v",
#         "copy",  # Copy the video stream without re-encoding
#         "-map",
#         "0:v:0",  # Map the video stream from the first input
#         "-map",
#         "0:a:0",  # Map the main audio stream from the first input
#         "-map",
#         "1:a:0",  # Map the background music stream from the second input
#         "-shortest",  # Finish encoding when the shortest input ends
#         "-y",  # Overwrite output file if it exists
#         output_path,  # Output file path
#     ]

#     # Execute the FFmpeg command
#     subprocess.run(command, check=True)


# # Example usage
# video_path = "../downloads/final_video_3.mp4"
# bg_music_path = "../downloads/bg_1.mp3"
# output_path = "../downloads/final_video_with_bg.mp4"

# # Adjust the bg_volume parameter to control the background music volume (e.g., 0.3 for 30% volume)
# if __name__ == "__main__":
#     add_background_music(video_path, bg_music_path, output_path, bg_volume=0.05)

# ------------------------

# import os
# import glob
# import multiprocessing
# import subprocess
# from dotenv import load_dotenv
# import os


# def add_background_music(video_file):
#     video_name = os.path.basename(video_file)
#     video_index = video_name.split("_")[-1].split(".")[0]  # Extract index from filename
#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Paths
#     bg_music_path = os.path.join(script_dir, "../downloads/bg_1.mp3")
#     output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")
#     output_path = os.path.join(output_dir, f"final_vid_with_bg_{video_index}.mp4")


#     os.makedirs(output_dir, exist_ok=True)

#     if not os.path.exists(bg_music_path):
#         print(f"Background music file not found for {video_name}, skipping...")
#         return

#     print(f"Adding background music to {video_name}...")
#     command = [
#         "ffmpeg",
#         "-i",
#         video_file,
#         "-i",
#         bg_music_path,
#         "-filter_complex",
#         "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=shortest",
#         "-c:v",
#         "copy",
#         "-map",
#         "0:v:0",
#         "-map",
#         "0:a:0",
#         "-map",
#         "1:a:0",
#         "-shortest",
#         "-y",
#         output_path,
#     ]

#     try:
#         subprocess.run(command, check=True)
#         print(f"Successfully created: {output_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error processing {video_name}: {e}")


# def process_background_music():
#     load_dotenv()

#     input_dir = os.path.join(output_dir, "../downloads")
#     video_files = sorted(glob.glob(os.path.join(input_dir, "final_video_*.mp4")))

#     if not video_files:
#         print(f"No matching video files found in {input_dir}")
#         return

#     print(f"Found {len(video_files)} videos to process")

#     # Use parallel processing to speed up the operation
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         pool.map(add_background_music, video_files)

#     print("All videos processed")


# if __name__ == "__main__":
#     process_background_music()





import os
import glob
import multiprocessing
import subprocess
from dotenv import load_dotenv


def add_background_music(video_file):
    """
    Add background music to a video file using ffmpeg.
    """
    # Extract video name and index
    video_name = os.path.basename(video_file)
    video_index = video_name.split("_")[-1].split(".")[0]  # Extract index from filename

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    bg_music_path = os.path.join(script_dir, "../downloads/bg_1.mp3")
    output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")
    output_path = os.path.join(output_dir, f"final_vid_with_bg_{video_index}.mp4")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if background music file exists
    if not os.path.exists(bg_music_path):
        print(f"Background music file not found at {bg_music_path}, skipping...")
        return

    print(f"Adding background music to {video_name}...")

    # FFmpeg command to add background music
    command = [
        "ffmpeg",
        "-i", video_file,  # Input video
        "-i", bg_music_path,  # Background music
        "-filter_complex", "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=shortest",  # Adjust volume and mix audio
        "-c:v", "copy",  # Copy video stream without re-encoding
        "-map", "0:v:0",  # Map video stream
        "-map", "0:a:0",  # Map original audio stream
        "-map", "1:a:0",  # Map background music stream
        "-shortest",  # Ensure output duration matches the shortest input
        "-y",  # Overwrite output file if it exists
        output_path,  # Output file
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Successfully created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_name}: {e}")


def process_background_music():
    """
    Process all videos in the downloads folder and add background music.
    """
    load_dotenv()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output directories
    input_dir = os.path.join(script_dir, "../downloads")
    output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")

    # Find all video files in the input directory
    video_files = sorted(glob.glob(os.path.join(input_dir, "final_video_*.mp4")))

    if not video_files:
        print(f"No matching video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to process")

    # Use parallel processing to speed up the operation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(add_background_music, video_files)

    print("All videos processed")


if __name__ == "__main__":
    process_background_music()