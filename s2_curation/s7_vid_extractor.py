# import os
# import json
# import subprocess

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Paths
# CURATED_SEGMENTS_PATH = os.path.join(script_dir, "assets/temp/curated_video_segments.json")
# INPUT_VIDEO_PATH = os.path.join(script_dir, "../downloads/input_video.mp4")
# OUTPUT_DIR = os.path.join(script_dir, "assets/curated_videos/")

# def extract_video_segments(
#     curated_segments_path=CURATED_SEGMENTS_PATH,
#     input_video_path=INPUT_VIDEO_PATH,
#     output_dir=OUTPUT_DIR,
# ):
#     """
#     Extract video segments using FFmpeg based on curated segments.

#     Args:
#         curated_segments_path (str): Path to JSON file with curated video segments
#         input_video_path (str): Path to the input video file
#         output_dir (str): Directory to save extracted video segments

#     Returns:
#         list: Paths of extracted video segments
#     """
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # List to store extracted video segment paths
#     extracted_segments = []

#     try:
#         # Read curated segments JSON
#         with open(curated_segments_path, "r") as file:
#             curated_segments = json.load(file)

#         # Process each segment
#         for idx, (transcript_filename, segment_info) in enumerate(
#             curated_segments.items(), 1
#         ):
#             # Extract start and end times
#             start_time = segment_info["start"]
#             end_time = segment_info["end"]

#             # Calculate duration
#             duration = end_time - start_time

#             # Output video path
#             output_video_path = os.path.join(output_dir, f"curated_vid_{idx}.mp4")

#             # Construct FFmpeg command - with changes to fix freezing issue
#             ffmpeg_command = [
#                 "ffmpeg",
#                 "-ss",
#                 str(start_time),  # Seek before input to avoid initial freeze
#                 "-i",
#                 input_video_path,
#                 "-t",
#                 str(duration),
#                 "-avoid_negative_ts",
#                 "make_zero",  # Helps with timestamp issues
#                 "-reset_timestamps",
#                 "1",  # Reset timestamps to avoid end freeze
#                 "-c:v",
#                 "libx264",  # Re-encode video
#                 "-c:a",
#                 "aac",  # Re-encode audio
#                 "-preset",
#                 "fast",  # Fast encoding preset
#                 "-crf",
#                 "22",  # Maintain good quality
#                 output_video_path,
#             ]

#             try:
#                 # Run FFmpeg command
#                 subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)

#                 # Add to extracted segments list
#                 extracted_segments.append(output_video_path)

#                 print(f"Extracted segment {idx}: {transcript_filename}")
#                 print(f"  Start: {start_time}s, End: {end_time}s")
#                 print(f"  Output: {output_video_path}")

#             except subprocess.CalledProcessError as e:
#                 print(f"Error extracting segment {idx}: {e}")
#                 print(f"FFmpeg error output: {e.stderr.decode()}")

#         return extracted_segments

#     except FileNotFoundError:
#         print(f"Error: Curated segments file not found at {curated_segments_path}")
#         return []
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON in {curated_segments_path}")
#         return []

# # Example usage
# if __name__ == "__main__":
#     extracted_videos = extract_video_segments()
#     print("\nExtracted Video Segments:")
#     for video in extracted_videos:
#         print(video)




#------------------------------------------ Parallel Code ------------------------------------------
import os
import json
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
CURATED_SEGMENTS_PATH = os.path.join(script_dir, "assets/temp/curated_video_segments.json")
INPUT_VIDEO_PATH = os.path.join(script_dir, "../downloads/input_video.mp4")
OUTPUT_DIR = os.path.join(script_dir, "assets/curated_videos/")

def extract_single_segment(segment_data):
    """
    Extract a single video segment using FFmpeg
    
    Args:
        segment_data (tuple): Contains (idx, transcript_filename, segment_info, input_video_path, output_dir)
        
    Returns:
        tuple: (success, output_path, error_message, idx, execution_time)
    """
    idx, transcript_filename, segment_info, input_video_path, output_dir = segment_data
    
    start_time = time.time()
    
    try:
        # Extract start and end times
        start_time_sec = segment_info["start"]
        end_time_sec = segment_info["end"]
        
        # Calculate duration
        duration = end_time_sec - start_time_sec
        
        # Output video path
        output_video_path = os.path.join(output_dir, f"curated_vid_{idx}.mp4")
        
        # Construct FFmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-ss", str(start_time_sec),
            "-i", input_video_path,
            "-t", str(duration),
            "-avoid_negative_ts", "make_zero",
            "-reset_timestamps", "1",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "22",
            "-y",  # Overwrite output files without asking
            output_video_path
        ]
        
        # Run FFmpeg command with stdout and stderr redirected
        process = subprocess.run(
            ffmpeg_command, 
            check=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        execution_time = time.time() - start_time
        return (True, output_video_path, None, idx, execution_time)
    
    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg error output: {e.stderr.decode()}"
        execution_time = time.time() - start_time
        return (False, None, error_message, idx, execution_time)
    
    except Exception as e:
        error_message = str(e)
        execution_time = time.time() - start_time
        return (False, None, error_message, idx, execution_time)


def extract_video_segments(
    curated_segments_path=CURATED_SEGMENTS_PATH,
    input_video_path=INPUT_VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    max_workers=None  # Default to number of CPU cores
):
    """
    Extract video segments in parallel using ProcessPoolExecutor
    
    Args:
        curated_segments_path (str): Path to JSON file with curated video segments
        input_video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted video segments
        max_workers (int): Maximum number of worker processes (default: CPU count)
        
    Returns:
        list: Paths of successfully extracted video segments
    """
    # Start timing
    start_time = time.time()
    
    # Set max_workers to CPU count if not specified
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Using {max_workers} parallel processes for video extraction")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store extracted video segment paths
    extracted_segments = []
    
    try:
        # Read curated segments JSON
        with open(curated_segments_path, "r") as file:
            curated_segments = json.load(file)
        
        # Prepare tasks for parallel processing
        tasks = []
        for idx, (transcript_filename, segment_info) in enumerate(curated_segments.items(), 1):
            task_data = (idx, transcript_filename, segment_info, input_video_path, output_dir)
            tasks.append(task_data)
        
        # Process segments in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(extract_single_segment, tasks))
        
        # Process results
        for success, output_path, error_message, idx, exec_time in results:
            if success:
                extracted_segments.append(output_path)
                print(f"Extracted segment {idx}: Duration: {exec_time:.2f}s")
                print(f"  Output: {output_path}")
            else:
                print(f"Error extracting segment {idx}: {error_message}")
        
        total_time = time.time() - start_time
        print(f"\nTotal extraction time: {total_time:.2f}s for {len(extracted_segments)} segments")
        
        return extracted_segments
    
    except FileNotFoundError:
        print(f"Error: Curated segments file not found at {curated_segments_path}")
        return []
    
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {curated_segments_path}")
        return []


# Example usage
if __name__ == "__main__":
    print("Starting parallel video extraction...")
    extracted_videos = extract_video_segments()
    
    print("\nExtracted Video Segments:")
    for video in extracted_videos:
        print(video)