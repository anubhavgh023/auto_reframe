# import os
# import glob
# import multiprocessing
# import subprocess
# from dotenv import load_dotenv


# def add_background_music(video_file):
#     """
#     Add background music to a video file using ffmpeg.
#     """
#     # Extract video name and index
#     video_name = os.path.basename(video_file)
#     video_index = video_name.split("_")[-1].split(".")[0]  # Extract index from filename

#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Define paths
#     bg_music_path = os.path.join(script_dir, "../downloads/bg_1.mp3")
#     output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")
#     output_path = os.path.join(output_dir, f"final_vid_with_bg_{video_index}.mp4")

#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Check if background music file exists
#     if not os.path.exists(bg_music_path):
#         print(f"Background music file not found at {bg_music_path}, skipping...")
#         return

#     print(f"Adding background music to {video_name}...")

#     # FFmpeg command to add background music
#     command = [
#         "ffmpeg",
#         "-i", video_file,  # Input video
#         "-i", bg_music_path,  # Background music
#         "-filter_complex", "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=shortest",  # Adjust volume and mix audio
#         "-c:v", "copy",  # Copy video stream without re-encoding
#         "-map", "0:v:0",  # Map video stream
#         "-map", "0:a:0",  # Map original audio stream
#         "-map", "1:a:0",  # Map background music stream
#         "-shortest",  # Ensure output duration matches the shortest input
#         "-y",  # Overwrite output file if it exists
#         output_path,  # Output file
#     ]

#     try:
#         # Run the FFmpeg command
#         subprocess.run(command, check=True)
#         print(f"Successfully created: {output_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error processing {video_name}: {e}")


# def process_background_music():
#     """
#     Process all videos in the downloads folder and add background music.
#     """
#     load_dotenv()

#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Define input and output directories
#     input_dir = os.path.join(script_dir, "../downloads")
#     output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")

#     # Find all video files in the input directory
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

# ----------------------------------------------
import os
import glob
import multiprocessing
import subprocess
from dotenv import load_dotenv


def add_background_music(video_file, bgm: str):
    """
    Add background music to a video file using ffmpeg.

    Args:
        video_file (str): Path to the input video file.
        bgm (str): Name of the background music file (without .mp3 extension).
    """
    # Extract video name and index
    video_name = os.path.basename(video_file)
    video_index = video_name.split("_")[-1].split(".")[0]  # Extract index from filename

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    bg_music_dir = os.path.join(script_dir, "./bg_music")
    bg_music_path = os.path.join(bg_music_dir, f"{bgm}.mp3")
    output_dir = os.path.join(script_dir, "../downloads/final_video_with_bg/")
    output_path = os.path.join(output_dir, f"final_vid_with_bg_{video_index}.mp4")

    # Create output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if background music file exists
    if not os.path.exists(bg_music_path):
        print(
            f"Background music file '{bgm}.mp3' not found at {bg_music_path}, skipping {video_name}..."
        )
        return

    print(f"Adding background music '{bgm}.mp3' to {video_name}...")

    # FFmpeg command to add background music
    command = [
        "ffmpeg",
        "-i",
        video_file,  # Input video
        "-i",
        bg_music_path,  # Background music
        "-filter_complex",
        "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=shortest",  # Adjust volume and mix audio
        "-c:v",
        "copy",  # Copy video stream without re-encoding
        "-map",
        "0:v:0",  # Map video stream
        "-map",
        "0:a:0",  # Map original audio stream
        "-map",
        "1:a:0",  # Map background music stream
        "-shortest",  # Ensure output duration matches the shortest input
        "-y",  # Overwrite output file if it exists
        output_path,  # Output file
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Successfully created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_name}: {e.stderr.decode()}")


def process_background_music(bgm: str):
    """
    Process all videos in the downloads folder and add specified background music.

    Args:
        bgm (str): Name of the background music file (without .mp3 extension, e.g., "time").
    """
    load_dotenv()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input directory
    input_dir = os.path.join(script_dir, "../downloads/final_videos")

    # Validate bgm parameter
    bg_music_dir = os.path.join(script_dir, "../bg_music")
    bg_music_path = os.path.join(bg_music_dir, f"{bgm}.mp3")
    if not os.path.exists(bg_music_path):
        print(
            f"Specified background music '{bgm}.mp3' not found in {bg_music_dir}. Available options:"
        )
        available_bgm = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(bg_music_dir, "*.mp3"))
        ]
        print(", ".join(available_bgm))
        return

    # Find all video files in the input directory
    video_files = sorted(glob.glob(os.path.join(input_dir, "final_video_*.mp4")))

    if not video_files:
        print(f"No matching video files found in {input_dir}")
        return

    print(
        f"Found {len(video_files)} videos to process with background music '{bgm}.mp3'"
    )

    # Use parallel processing to speed up the operation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            add_background_music, [(video_file, bgm) for video_file in video_files]
        )

    print("All videos processed")


if __name__ == "__main__":
    # Example usage: process_background_music("time")
    process_background_music("time")
