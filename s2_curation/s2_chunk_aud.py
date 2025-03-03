import subprocess
import os
import math

# Global variables
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global variables
INPUT_AUDIO = os.path.join(
    script_dir, "assets/audio/full_audio.wav"
)  # Path to full audio file
OUTPUT_DIR = os.path.join(
    script_dir, "assets/audio/chunks/"
)  # Directory to save chunks


def chunk_audio(chunk_duration: int):
    """
    Splits the global audio file into smaller WAV chunks of specified duration.

    :param chunk_duration: Duration of each chunk in seconds.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get total duration of the audio file using FFprobe
    command_duration = [
        "ffprobe",
        "-i",
        INPUT_AUDIO,
        "-show_entries",
        "format=duration",
        "-v",
        "quiet",
        "-of",
        "csv=p=0",
    ]
    total_duration = float(subprocess.check_output(command_duration).decode().strip())

    # Calculate number of chunks (ignoring leftover audio)
    num_chunks = (
        total_duration // chunk_duration
    )  # Floor division ensures ignoring extra time

    # Split the audio into chunks
    for i in range(int(num_chunks)):  # Ensure integer loop count
        start_time = i * chunk_duration
        output_file = os.path.join(OUTPUT_DIR, f"chunk_{i+1}.wav")

        command_split = [
            "ffmpeg",
            "-i",
            INPUT_AUDIO,
            "-ss",
            str(start_time),  # Start time
            "-t",
            str(chunk_duration),  # Duration
            "-acodec",
            "pcm_s16le",  # WAV format
            "-ar",
            "44100",  # Sample rate
            "-ac",
            "2",  # Stereo
            "-y",
            output_file,  # Overwrite if exists
        ]

        subprocess.run(command_split, check=True)
        print(f"Chunk {i+1} saved: {output_file}")


# Example usage
if __name__ == "__main__":
    chunk_audio(60)  # Splits the audio into 30-second WAV chunks, ignoring leftover audio
