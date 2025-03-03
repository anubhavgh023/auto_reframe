import os
import random
import subprocess
import shlex
from typing import List

def get_video_duration(input_video: str) -> float:
    """Get total video duration in seconds."""
    probe_cmd = shlex.split(
        f"ffprobe -v error -show_entries format=duration "
        f'-of default=noprint_wrappers=1:nokey=1 "{input_video}"'
    )
    return float(subprocess.check_output(probe_cmd).decode().strip())

def select_strategic_segments(total_segments: int, num_select: int = 4) -> List[int]:
    """
    Select segments strategically spread across the video.
    
    Args:
        total_segments (int): Total number of segments
        num_select (int): Number of segments to select
    
    Returns:
        List of selected segment indices
    """
    # Set a fixed seed for reproducibility
    random.seed(42)
    
    # Create segments list
    all_segments = list(range(total_segments))
    
    # If total segments is less than requested, return all segments
    if total_segments <= num_select:
        return all_segments
    
    # Strategic selection of segments
    selected = []
    step = total_segments // (num_select + 1)
    
    for i in range(num_select):
        # Base selection with some randomness
        base_index = (i + 1) * step
        
        # Add some randomness within a small window
        variation = random.randint(-step//4, step//4)
        selected_segment = min(max(0, base_index + variation), total_segments - 1)
        
        selected.append(selected_segment)
    
    return sorted(selected)

def extract_and_speed_up_segments(
    input_video: str, 
    segments: List[int], 
    segment_duration: int = 60, 
    output_dir: str = "curated_videos"
) -> List[str]:
    """
    Extract and speed up selected video segments.
    
    Args:
        input_video (str): Path to input video
        segments (List[int]): List of segment indices to extract
        segment_duration (int): Duration of each segment
        output_dir (str): Directory to save curated videos
    
    Returns:
        List of curated video file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    curated_videos = []
    
    for i, segment_index in enumerate(segments, 1):
        start_time = segment_index * segment_duration
        output_video = os.path.join(output_dir, f"curated_video_{i:02d}.mp4")
        
        # FFmpeg command to extract segment and speed up
        cmd = shlex.split(
        f'ffmpeg -i "{input_video}" '
        f"-ss {start_time} "
        f"-t {segment_duration} "
        f'-vf "setpts=0.8*PTS" '  # Speed up video 
        f'-af "atempo=1.25" '     # Speed up audio
        f"-c:v libx264 -preset fast "
        f"-c:a aac "
        f'"{output_video}"'
        )
        
        subprocess.run(cmd, check=True, capture_output=True)
        curated_videos.append(output_video)
        print(f"Created: {output_video}")
    
    return curated_videos

def main():
    # Hardcoded input video path - REPLACE WITH YOUR ACTUAL PATH
    input_video = "../downloads/input_video.mp4"
    
    # Get video duration
    total_duration = int(get_video_duration(input_video))
    
    # Calculate number of 1-minute segments
    segment_duration = 60  # 1 minute
    total_segments = total_duration // segment_duration
    
    # Select strategic segments
    selected_segments = select_strategic_segments(total_segments)
    
    print("Selected segment indices:", selected_segments)
    
    # Extract and speed up segments
    curated_videos = extract_and_speed_up_segments(
        input_video, 
        selected_segments, 
        segment_duration=segment_duration,
        output_dir="../downloads/curated_videos"
    )
    
    print("\nCurated videos created:")
    for video in curated_videos:
        print(f"- {video}")

if __name__ == "__main__":
    main()