import os
import json
from datetime import datetime
import time
from tqdm import tqdm

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
TRANSCRIPTS_DIR = os.path.join(script_dir, "assets/audio/transcript_chunks/")
SLIDING_TRANSCRIPTS_DIR = os.path.join(
    script_dir, "assets/audio/sliding_transcript_chunks/"
)

# Ensure sliding transcript directory exists
os.makedirs(SLIDING_TRANSCRIPTS_DIR, exist_ok=True)

# Window size in seconds
WINDOW_SIZE = 60


def read_all_transcripts(transcript_dir=TRANSCRIPTS_DIR):
    """Read all transcript files and return them as a sorted list."""
    transcript_files = sorted(
        [
            f
            for f in os.listdir(transcript_dir)
            if f.startswith("transcript_chunk_") and f.endswith(".json")
        ],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    all_transcripts = []
    for transcript_file in transcript_files:
        file_path = os.path.join(transcript_dir, transcript_file)
        try:
            with open(file_path, "r") as file:
                transcript = json.load(file)
                all_transcripts.append(transcript)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {transcript_file}: {e}")

    return all_transcripts


def get_all_words(transcripts):
    """Extract all words with their timestamps from all transcripts."""
    all_words = []
    for transcript in transcripts:
        if "words" in transcript:
            all_words.extend(transcript["words"])

    # Sort words by start time
    all_words.sort(key=lambda x: x["start"])
    return all_words


def create_sliding_windows(all_words, window_size=WINDOW_SIZE, step_size=10):
    """Create sliding windows of words."""
    if not all_words:
        return []

    # Get the time range of all words
    start_time = all_words[0]["start"]
    end_time = all_words[-1]["end"]

    windows = []
    window_start = start_time

    while window_start < end_time:
        window_end = window_start + window_size

        # Filter words that fall within this window
        window_words = [
            word
            for word in all_words
            if word["start"] >= window_start and word["end"] <= window_end
        ]

        # If no complete words fit in the window but there are words that overlap,
        # include words that at least start within the window
        if not window_words:
            window_words = [
                word
                for word in all_words
                if word["start"] >= window_start and word["start"] < window_end
            ]

        if window_words:
            # Create the window transcript
            window_transcript = {
                "window_start": window_start,
                "window_end": window_end,
                "text": " ".join([w["word"] for w in window_words]),
                "words": window_words,
            }
            windows.append(window_transcript)

        # Move window forward
        window_start += step_size

    return windows


def save_sliding_windows(windows, output_dir=SLIDING_TRANSCRIPTS_DIR):
    """Save each window as a separate JSON file."""
    for i, window in enumerate(windows, start=1):
        output_path = os.path.join(output_dir, f"sliding_transcript_{i}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(window, f, indent=4)

    return len(windows)


def generate_sliding_transcripts():
    """Main function to generate sliding window transcripts."""
    print("Reading transcript chunks...")
    all_transcripts = read_all_transcripts()

    if not all_transcripts:
        print("No transcript chunks found!")
        return False

    print(f"Found {len(all_transcripts)} transcript chunks")

    print("Extracting all words...")
    all_words = get_all_words(all_transcripts)

    print(f"Creating sliding windows (size: {WINDOW_SIZE}s)...")
    windows = create_sliding_windows(all_words)

    print(f"Saving {len(windows)} sliding window transcripts...")
    num_saved = save_sliding_windows(windows)

    print(f"\nSliding window generation completed!")
    print(f"Total sliding windows created: {num_saved}")

    return True


if __name__ == "__main__":
    start_time = time.time()
    generate_sliding_transcripts()
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
