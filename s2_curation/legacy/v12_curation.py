import os
import random
import subprocess
import shlex
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_video_duration(input_video: str) -> float:
    """Get total video duration in seconds."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")

    probe_cmd = shlex.split(
        f"ffprobe -v error -show_entries format=duration "
        f'-of default=noprint_wrappers=1:nokey=1 "{input_video}"'
    )
    try:
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Video duration: {duration} seconds")
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        raise


def select_strategic_segments(total_segments: int, num_select: int = 4) -> List[int]:
    """
    Select segments strategically spread across the video, avoiding the first segment
    and ensuring good distribution.
    """
    print(f"Selecting {num_select} segments from {total_segments} total segments")

    if total_segments <= 0:
        print("Warning: No segments available")
        return []

    if total_segments <= 1:
        return [0]

    if total_segments <= num_select:
        return list(range(1, total_segments))

    # Ensure we don't select more segments than available (excluding first segment)
    num_select = min(num_select, total_segments - 1)

    # Define regions for segment selection
    regions = []
    usable_segments = total_segments - 1  # Exclude first segment
    region_size = usable_segments / num_select

    for i in range(num_select):
        region_start = 1 + int(i * region_size)
        region_end = 1 + int((i + 1) * region_size)
        regions.append((region_start, region_end))

    selected = []
    for start, end in regions:
        variation_range = max(1, int((end - start) * 0.2))
        base_pos = (start + end) // 2
        variation = random.randint(-variation_range, variation_range)
        selected_pos = max(1, min(total_segments - 1, base_pos + variation))
        selected.append(selected_pos)

    # Ensure no duplicates and maintain order
    selected = sorted(list(set(selected)))

    # Fill gaps if needed
    while len(selected) < num_select and len(selected) < total_segments - 1:
        gaps = [
            (b - a, a, b) for a, b in zip([0] + selected, selected + [total_segments])
        ]
        largest_gap = max(gaps)
        new_segment = (largest_gap[1] + largest_gap[2]) // 2
        if new_segment not in selected and new_segment > 0:
            selected.append(new_segment)
            selected.sort()

    print(f"Selected segments: {selected}")
    return selected


def process_single_segment(args) -> str:
    """Process a single video segment."""
    input_video, segment_index, segment_duration, output_dir, i = args
    start_time = segment_index * segment_duration
    output_video = os.path.join(output_dir, f"curated_video_{i:02d}.mp4")

    cmd = shlex.split(
        f'ffmpeg -i "{input_video}" '
        f"-ss {start_time} "
        f"-t {segment_duration} "
        f'-filter_complex "[0:v]setpts=0.8*PTS[v];[0:a]atempo=1.25[a]" '
        f'-map "[v]" -map "[a]" '
        f"-c:v libx264 -preset fast "
        f"-c:a aac "
        f"-t 45 "
        f'"{output_video}"'
    )

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully processed segment {i} at index {segment_index}")
        return output_video
    except subprocess.CalledProcessError as e:
        print(f"Error processing segment {i}: {e}")
        raise


def extract_and_speed_up_segments(
    input_video: str,
    segments: List[int],
    segment_duration: int = 60,
    output_dir: str = "curated_videos",
    max_workers: int = None,
) -> List[str]:
    """Extract and speed up selected video segments in parallel."""
    if not segments:
        raise ValueError("No segments provided for processing")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(segments)} segments in parallel")

    process_args = [
        (input_video, segment_index, segment_duration, output_dir, i)
        for i, segment_index in enumerate(segments, 1)
    ]

    curated_videos = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(process_single_segment, args): i
            for i, args in enumerate(process_args, 1)
        }

        for future in as_completed(future_to_segment):
            segment_num = future_to_segment[future]
            try:
                output_video = future.result()
                curated_videos.append(output_video)
                print(f"Completed segment {segment_num}")
            except Exception as e:
                print(f"Failed to process segment {segment_num}: {str(e)}")

    return sorted(curated_videos)


def main():
    # Input validation
    input_video = "../downloads/input_video.mp4"
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Get video duration with error handling
    try:
        total_duration = int(get_video_duration(input_video))
        print(f"Total video duration: {total_duration} seconds")

        if total_duration <= 0:
            raise ValueError("Invalid video duration")

        # Calculate segments
        segment_duration = 60  # 1 minute
        total_segments = total_duration // segment_duration
        print(f"Total number of segments: {total_segments}")

        if total_segments <= 0:
            raise ValueError("Video is too short to segment")

        # Select segments
        selected_segments = select_strategic_segments(total_segments)
        if not selected_segments:
            raise ValueError("No segments were selected")

        print(f"Selected segment indices: {selected_segments}")

        # Process segments
        output_dir = "../downloads/curated_videos"
        curated_videos = extract_and_speed_up_segments(
            input_video,
            selected_segments,
            segment_duration=segment_duration,
            output_dir=output_dir,
        )

        print("\nCurated videos created:")
        for video in curated_videos:
            print(f"- {video}")

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
