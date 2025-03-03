import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import shlex
import os
import subprocess

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
INPUT_VIDEO = "../downloads/input_video.mp4"
AUDIO_FILE = "../downloads/extracted_audio.wav"
OUTPUT_DIR = "../downloads"
CURATED_VIDEO = os.path.join(OUTPUT_DIR, "curated_video.mp4")


# working
def extract_audio(input_video, output_audio):
    """Extract audio from video using ffmpeg, using shlex for command construction."""
    try:
        cmd = f'ffmpeg -i {input_video} -q:a 0 -map a {output_audio}' 
        cmd_list = shlex.split(cmd)  # Split the command string into a list

        subprocess.run(
            cmd_list, check=True, capture_output=True, text=True
        )  # changed the cmd to cmd_list
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction error: {e.stderr}")
        raise


# working
def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        return transcript.words
    except Exception as e:
        print(f"Transcription error: {e}")
        raise

import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def find_best_segments(words, target_duration=45):  # Target duration in seconds
    """
    Finds multiple engaging segments in a transcript using GPT-4, aiming for a total duration close to target_duration.

    Args:
        words: A list of word objects with start and end timestamps.
        target_duration: The desired total duration of the selected segments (in seconds).

    Returns:
        A list of tuples, where each tuple represents a segment (start_time, end_time).
        Returns an empty list if no suitable segments are found.
    """
    try:
        transcript_text = "\n".join(
            [f"{word.start:.2f}-{word.end:.2f}: {word.word}" for word in words]
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""Analyze the following transcript to identify multiple engaging segments that, when combined, 
                    have a total duration of approximately {target_duration} seconds.  Prioritize segments with:

                    1.  Strong narrative hooks
                    2.  High emotional or informative value
                    3.  Clear context and minimal overlap

                    Return ONLY a JSON array of segments. Each segment should be a JSON object with "start" and "end" keys 
                    representing the start and end times in seconds. Ensure the total duration is close to {target_duration} seconds.

                    Example:
                    [
                      {{"start": 10.50, "end": 15.25}},
                      {{"start": 22.10, "end": 30.80}},
                      {{"start": 40.00, "end": 48.50}}
                    ]""",
                },
                {"role": "user", "content": transcript_text},
            ],
            response_format={"type": "json_object"},
        )

        try:
            segments_json = json.loads(response.choices[0].message.content)

            # Basic validation:
            if not isinstance(segments_json, list):
                print("Error: GPT-4 returned a non-list JSON object.")
                return []

            segments = []
            for seg in segments_json:
                if isinstance(seg, dict) and "start" in seg and "end" in seg:
                    start = float(seg["start"])
                    end = float(seg["end"])
                    segments.append((start, end))
                else:
                    print(f"Warning: Invalid segment format: {seg}")

            return segments

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing GPT-4 response: {e}")
            return []

    except Exception as e:
        print(f"Segment selection error: {e}")
        return []



def main():
    # Extract audio
    audio_path = extract_audio(INPUT_VIDEO, AUDIO_FILE)
    
    # Transcribe audio
    words = transcribe_audio(AUDIO_FILE)
    print(words)
    print("\n-------------\n")
    
    # Find best segment (using first target duration)
    best_parts = find_best_segments(words,45)
    print(best_parts)

if __name__ == "__main__":
    main()
