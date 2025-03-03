import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import shlex
import subprocess
import spacy
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from textblob import TextBlob
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced paths with error checking
INPUT_VIDEO = "../downloads/input_video.mp4"
AUDIO_FILE = "../downloads/extracted_audio.wav"
OUTPUT_DIR = "../downloads"
CURATED_VIDEO = os.path.join(OUTPUT_DIR, "curated_video.mp4")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize NLP tools
try:
    nltk.data.find("sentiment/vader_lexicon")
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("averaged_perceptron_tagger")
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download("vader_lexicon")
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()


class VideoSegment:
    def __init__(self, start, end, text, score=0):
        self.start = start
        self.end = end
        self.text = text
        self.score = score
        self.duration = end - start

    def __repr__(self):
        return f"Segment({self.start:.2f}-{self.end:.2f}, score={self.score:.2f})"


def extract_audio(input_video, output_audio):
    """Extract audio with enhanced error handling and quality settings."""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_video,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # High quality audio codec
            "-ar",
            "44100",  # Standard sample rate
            "-ac",
            "2",  # Stereo
            "-y",  # Overwrite output
            output_audio,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        return output_audio

    except subprocess.CalledProcessError as e:
        print(f"Audio extraction error: {e.stderr}")
        raise


def transcribe_audio(audio_path):
    """Transcribe audio with enhanced error handling and retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            return transcript.words
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Transcription attempt {attempt + 1} failed: {e}")
            continue


def calculate_engagement_score(text):
    """Calculate an engagement score based on multiple factors."""
    # Sentiment analysis
    sentiment_score = analyzer.polarity_scores(text)["compound"]

    # Question detection
    question_bonus = 2 if "?" in text else 0

    # Emotion words detection
    blob = TextBlob(text)
    emotion_words = sum(1 for word, tag in blob.tags if tag.startswith("JJ"))

    # Key phrase detection
    key_phrases = ["how to", "why", "best", "top", "secret", "never", "always"]
    phrase_bonus = sum(3 for phrase in key_phrases if phrase in text.lower())

    # Calculate final score
    engagement_score = (
        abs(sentiment_score) * 2 + question_bonus + emotion_words + phrase_bonus
    )

    return engagement_score


def find_best_segments(words, target_duration=45, min_segment_duration=5):
    """Find best segments with improved selection criteria."""
    segments = []
    current_segment = []
    current_text = []

    for word in words:
        current_segment.append(word)
        current_text.append(word.word)

        # Check if we have enough words for a meaningful segment
        if len(current_segment) >= 10:
            segment_text = " ".join(current_text)

            # Calculate segment duration
            start_time = current_segment[0].start
            end_time = current_segment[-1].end
            duration = end_time - start_time

            # Only consider segments that are long enough
            if duration >= min_segment_duration:
                # Calculate engagement score
                engagement_score = calculate_engagement_score(segment_text)

                # Create segment object
                segment = VideoSegment(
                    start=start_time,
                    end=end_time,
                    text=segment_text,
                    score=engagement_score,
                )

                segments.append(segment)

            # Reset for next segment with overlap
            overlap = 5  # Number of words to overlap
            current_segment = current_segment[-overlap:]
            current_text = current_text[-overlap:]

    # Sort segments by score
    segments.sort(key=lambda x: x.score, reverse=True)

    # Select best segments that fit within target duration
    selected_segments = []
    total_duration = 0

    for segment in segments:
        if total_duration + segment.duration <= target_duration:
            selected_segments.append(segment)
            total_duration += segment.duration

    # Sort selected segments by start time
    selected_segments.sort(key=lambda x: x.start)

    return [(segment.start, segment.end) for segment in selected_segments]


def crop_video(input_video, output_video, segments):
    """Crop video with enhanced quality settings and proper path handling."""
    try:
        # Create temporary directory for segment files
        temp_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "temp", "segments"))
        os.makedirs(temp_dir, exist_ok=True)

        # Extract each segment to a separate file
        segment_files = []
        for i, (start, end) in enumerate(segments):
            segment_file = os.path.join(temp_dir, f"segment_{i}.mp4")
            segment_files.append(segment_file)

            # Construct ffmpeg command for segment extraction
            cmd = shlex.split(
                f'ffmpeg -i "{input_video}" '
                f"-ss {start:.3f} "
                f"-t {end-start:.3f} "
                f"-c:v libx264 -preset slow "
                f"-crf 18 "
                f"-c:a aac -b:a 192k "
                f"-avoid_negative_ts 1 "
                f'"{segment_file}"'
            )

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Verify segment file was created
            if not os.path.exists(segment_file):
                raise FileNotFoundError(
                    f"Failed to create segment file: {segment_file}"
                )

        # Create concat file with absolute paths
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for segment_file in segment_files:
                # Use forward slashes for ffmpeg compatibility
                segment_path = segment_file.replace("\\", "/")
                f.write(f"file '{segment_path}'\n")

        # Verify concat file was created and has content
        if not os.path.exists(concat_file):
            raise FileNotFoundError(f"Failed to create concat file: {concat_file}")

        # Construct ffmpeg command for concatenation
        concat_cmd = shlex.split(
            f"ffmpeg -f concat -safe 0 "
            f'-i "{concat_file}" '
            f"-c copy "
            f'"{output_video}"'
        )

        # Execute concatenation
        result = subprocess.run(concat_cmd, check=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg stderr output: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, concat_cmd, result.stderr
            )

        return output_video

    except subprocess.CalledProcessError as e:
        print(f"Video cropping error: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        raise
    # finally:
    #     # Cleanup temporary files
    #     try:
    #         if os.path.exists(temp_dir):
    #             import shutil

    #             shutil.rmtree(temp_dir)
    #     except Exception as e:
    #         print(f"Warning: Failed to clean up temporary files: {str(e)}")


def main():
    print("Starting video curation process...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Extract audio
        print("Extracting audio...")
        audio_path = extract_audio(INPUT_VIDEO, AUDIO_FILE)

        # Transcribe audio
        print("Transcribing audio...")
        words = transcribe_audio(audio_path)

        # Find best segments
        print("Analyzing content and finding best segments...")
        best_segments = find_best_segments(
            words, target_duration=45, min_segment_duration=5
        )

        # Crop and save video
        print("Creating final video...")
        crop_video(INPUT_VIDEO, CURATED_VIDEO, best_segments)

        print(f"Curated video successfully saved to: {CURATED_VIDEO}")

        # Print segment information
        print("\nSelected segments:")
        for i, (start, end) in enumerate(best_segments, 1):
            duration = end - start
            print(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {duration:.2f}s)")

    except Exception as e:
        print(f"Error during video curation: {str(e)}")
        raise
    # finally:
    #     # Cleanup temporary files
    #     if os.path.exists(AUDIO_FILE):
    #         os.remove(AUDIO_FILE)


if __name__ == "__main__":
    main()
