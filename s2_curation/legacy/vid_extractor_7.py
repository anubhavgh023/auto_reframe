import os
import json
import subprocess


def extract_video_segments(
    curated_segments_path="./assets/temp/curated_video_segments.json",
    input_video_path="../downloads/input_video.mp4",
    output_dir="./assets/curated_videos/",
):
    """
    Extract video segments using FFmpeg based on curated segments.

    Args:
        curated_segments_path (str): Path to JSON file with curated video segments
        input_video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted video segments

    Returns:
        list: Paths of extracted video segments
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List to store extracted video segment paths
    extracted_segments = []

    try:
        # Read curated segments JSON
        with open(curated_segments_path, "r") as file:
            curated_segments = json.load(file)

        # Process each segment
        for idx, (transcript_filename, segment_info) in enumerate(
            curated_segments.items(), 1
        ):
            # Extract start and end times
            start_time = segment_info["start"]
            end_time = segment_info["end"]

            # Calculate duration
            duration = end_time - start_time

            # Output video path
            output_video_path = os.path.join(output_dir, f"curated_vid_{idx}.mp4")

            # Construct FFmpeg command
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                input_video_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",  # Use stream copy to avoid re-encoding
                output_video_path,
            ]

            try:
                # Run FFmpeg command
                subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)

                # Add to extracted segments list
                extracted_segments.append(output_video_path)

                print(f"Extracted segment {idx}: {transcript_filename}")
                print(f"  Start: {start_time}s, End: {end_time}s")
                print(f"  Output: {output_video_path}")

            except subprocess.CalledProcessError as e:
                print(f"Error extracting segment {idx}: {e}")
                print(f"FFmpeg error output: {e.stderr.decode()}")

        return extracted_segments

    except FileNotFoundError:
        print(f"Error: Curated segments file not found at {curated_segments_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {curated_segments_path}")
        return []


# Example usage
if __name__ == "__main__":
    extracted_videos = extract_video_segments()
    print("\nExtracted Video Segments:")
    for video in extracted_videos:
        print(video)
