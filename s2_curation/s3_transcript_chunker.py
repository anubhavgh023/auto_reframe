import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import time
from functools import partial
from tqdm import tqdm

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global variables
CHUNKS_DIR = os.path.join(script_dir, "assets/audio/chunks/")
TRANSCRIPTS_DIR = os.path.join(script_dir, "assets/audio/transcript_chunks/")

# Ensure transcript directory exists
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


def process_single_file(api_key: str, file_tuple: tuple) -> dict:
    """Process a single audio file and return its transcript data."""
    idx, file_name = file_tuple
    audio_path = os.path.join(CHUNKS_DIR, file_name)
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"transcript_chunk_{idx}.json")

    client = OpenAI(api_key=api_key)
    result = {"success": False, "file": file_name, "index": idx, "error": None}

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        transcript_data = {
            "duration": transcript.duration,
            "language": transcript.language,
            "text": transcript.text,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in transcript.words
            ],
        }

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=4)

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def adjust_transcript_timestamps(transcript_dir=TRANSCRIPTS_DIR, chunk_duration=60):
    """
    Adjust transcript timestamps to sync with video chunks.
    """
    print("\nAdjusting timestamps...")

    # Get all transcript files sorted by their chunk number
    transcript_files = sorted(
        [
            f
            for f in os.listdir(transcript_dir)
            if f.startswith("transcript_chunk_") and f.endswith(".json")
        ],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    for transcript_file in transcript_files:
        # Extract chunk number
        chunk_number = int(transcript_file.split("_")[-1].split(".")[0])

        # Calculate time offset
        time_offset = (chunk_number - 1) * chunk_duration

        # Full path to the transcript file
        file_path = os.path.join(transcript_dir, transcript_file)

        try:
            # Read the transcript file
            with open(file_path, "r") as file:
                transcript = json.load(file)

            # Adjust timestamps for each word
            for word in transcript["words"]:
                word["start"] += time_offset
                word["end"] += time_offset

            # Write back the modified transcript
            with open(file_path, "w") as file:
                json.dump(transcript, file, indent=4)

            print(f"Adjusted timestamps for {transcript_file}")

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing {transcript_file}: {e}")


def process_chunks():
    """Process all audio chunks and generate transcripts."""
    # Get list of audio files
    audio_files = sorted(
        [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),  # Sort by chunk number
    )
    total_files = len(audio_files)

    print(f"\nStarting transcription of {total_files} files...")
    print("Maximum concurrent processes: 5")

    # Create list of tuples containing index and filename
    files_with_indices = list(enumerate(audio_files, start=1))

    # Create a partial function with the API key
    process_func = partial(process_single_file, API_KEY)

    start_time = time.time()
    successful = 0
    failed = 0

    # Process files using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Use tqdm for progress bar
        results = list(
            tqdm(
                executor.map(process_func, files_with_indices),
                total=len(files_with_indices),
                desc="Processing files",
                unit="file",
            )
        )

    # Calculate statistics
    for result in results:
        if result["success"]:
            successful += 1
        else:
            failed += 1
            print(f"\nFailed to process {result['file']}: {result['error']}")

    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nTranscription completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per file: {total_time/total_files:.1f} seconds")
    print(f"Successful: {successful}/{total_files}")
    print(f"Failed: {failed}/{total_files}")

    return successful > 0  # Return True if at least one file was successful


def main():
    """Main execution function."""
    # First process all chunks
    if process_chunks():
        # Only adjust timestamps if processing was successful
        adjust_transcript_timestamps()
        print("\nAll processing completed!")
    else:
        print("\nNo successful transcriptions to adjust timestamps for.")


if __name__ == "__main__":
    main()
