import subprocess
import os


def extract_audio(input_video: str, output_audio: str):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    # FFmpeg command to extract audio
    command = [
        "ffmpeg",
        "-i",
        input_video,  # Input video
        "-q:a",
        "0",  # Highest audio quality
        "-map",
        "a",  # Extract only audio
        "-y",
        output_audio,  # Output audio file
    ]

    # Run the command
    subprocess.run(command, check=True)
    print(f"Audio extracted successfully: {output_audio}")


# Example usage
if __name__ == "__main__":
    input_video_path = "../downloads/input_video.mp4"
    extract_audio(input_video_path, "./assets/audio/full_audio.wav")
