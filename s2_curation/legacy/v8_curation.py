import os
import json
import random
import numpy as np
import subprocess
import shlex
import argparse
from typing import List, Tuple, Dict, Any

# External library imports
from openai import OpenAI
from dotenv import load_dotenv
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure paths
INPUT_VIDEO = os.path.abspath("../downloads/input_video.mp4")
OUTPUT_DIR = os.path.abspath("../downloads/curated")
TEMP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "temp"))

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# NLTK and Spacy initialization
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class VideoSegment:
    def __init__(self, start: float, end: float, text: str = "", score: float = 0):
        self.start = start
        self.end = end
        self.text = text
        self.score = score
        self.duration = end - start

    def __repr__(self):
        return f"Segment({self.start:.2f}-{self.end:.2f}, score={self.score:.2f})"


def extract_audio(input_video: str, output_audio: str) -> str:
    """Extract audio with enhanced error handling."""
    try:
        cmd = shlex.split(
            f'ffmpeg -i "{input_video}" '
            f"-vn -acodec pcm_s16le "
            f"-ar 44100 -ac 2 "
            f'-y "{output_audio}"'
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        return output_audio

    except subprocess.CalledProcessError as e:
        print(f"Audio extraction error: {e.stderr}")
        raise


def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """Transcribe audio with error handling and retry logic."""
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


def calculate_engagement_score(text: str) -> float:
    """Calculate an engagement score with multiple metrics."""
    # Sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]

    # Question detection
    question_bonus = 3 if "?" in text else 0

    # Emotion and descriptive words
    blob = TextBlob(text)
    emotion_words = sum(1 for word, tag in blob.tags if tag.startswith("JJ"))

    # Key phrase detection
    key_phrases = [
        "how to",
        "why",
        "best",
        "top",
        "secret",
        "never",
        "always",
        "must",
        "amazing",
        "incredible",
        "important",
        "fascinating",
        "learn",
        "discover",
        "understand",
        "explain",
    ]
    phrase_bonus = sum(4 for phrase in key_phrases if phrase in text.lower())

    # Spacy entity recognition for substantive content
    doc = nlp(text)
    entity_bonus = len(doc.ents) * 2

    # Length consideration
    length_bonus = len(text.split()) / 10

    # Comprehensive score calculation
    engagement_score = (
        abs(sentiment_score) * 3
        + question_bonus
        + emotion_words * 1.5
        + phrase_bonus
        + entity_bonus
        + length_bonus
    )

    return engagement_score


def crop_video(
    input_video: str, output_video: str, segments: List[Tuple[float, float]]
) -> str:
    """Crop video with enhanced quality settings."""
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

        # Create concat file
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for segment_file in segment_files:
                segment_path = segment_file.replace("\\", "/")
                f.write(f"file '{segment_path}'\n")

        # Construct ffmpeg command for concatenation
        concat_cmd = shlex.split(
            f"ffmpeg -f concat -safe 0 "
            f'-i "{concat_file}" '
            f"-c copy "
            f'"{output_video}"'
        )

        # Execute concatenation
        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)

        return output_video

    except subprocess.CalledProcessError as e:
        print(f"Video cropping error: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        raise


class AdvancedVideoAnalyzer:
    def __init__(self, input_video: str):
        self.input_video = input_video
        self.total_duration = self.get_video_duration()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def get_video_duration(self) -> float:
        """Get total video duration in seconds."""
        probe_cmd = shlex.split(
            f"ffprobe -v error -show_entries format=duration "
            f'-of default=noprint_wrappers=1:nokey=1 "{self.input_video}"'
        )
        return float(subprocess.check_output(probe_cmd).decode().strip())

    def sample_video_segments(
        self, num_segments: int = 6, segment_duration: int = 60
    ) -> List[Tuple[str, float, float]]:
        """Sample random non-overlapping segments from the video."""
        segments = []
        sampled_starts = set()

        max_start = max(0, self.total_duration - segment_duration)

        while len(segments) < num_segments:
            start = random.uniform(0, max_start)

            if any(
                abs(start - prev_start) < segment_duration
                for prev_start in sampled_starts
            ):
                continue

            end = min(start + segment_duration, self.total_duration)
            segment_file = os.path.join(
                TEMP_DIR, f"sample_segment_{len(segments):02d}.mp4"
            )

            # Extract segment
            extract_cmd = shlex.split(
                f'ffmpeg -i "{self.input_video}" '
                f"-ss {start} "
                f"-t {end - start} "
                f"-c:v libx264 -preset fast "
                f"-c:a aac "
                f'"{segment_file}"'
            )
            subprocess.run(extract_cmd, check=True, capture_output=True)

            segments.append((segment_file, start, end))
            sampled_starts.add(start)

        return segments

    def analyze_segment_importance(self, segment_file: str) -> float:
        """Analyze segment importance using advanced NLP techniques."""
        # Extract audio
        segment_audio = os.path.join(
            TEMP_DIR, f"audio_{os.path.basename(segment_file)}.wav"
        )
        extract_audio(segment_file, segment_audio)

        try:
            # Transcribe audio
            words = transcribe_audio(segment_audio)

            # Combine words into full text
            full_text = " ".join(word.word for word in words)

            # Multiple importance metrics
            sentiment_score = abs(
                self.sentiment_analyzer.polarity_scores(full_text)["compound"]
            )

            # Identify key entities and concepts
            doc = nlp(full_text)
            named_entities = len(doc.ents)

            # Check for key action words and questions
            action_words = ["explain", "how", "why", "best", "important", "key"]
            action_word_count = sum(
                1 for word in action_words if word in full_text.lower()
            )

            # Combine metrics with weighted scoring
            importance_score = (
                sentiment_score * 2  # Strong emphasis on sentiment intensity
                + named_entities * 0.5  # Entities indicate substantive content
                + action_word_count * 0.7  # Action-oriented language
            )

            return importance_score

        except Exception as e:
            print(f"Error analyzing segment {segment_file}: {e}")
            return 0
        finally:
            # Clean up temporary audio
            if os.path.exists(segment_audio):
                os.remove(segment_audio)

    def curate_top_segments(
        self, num_curated: int = 4, target_duration: int = 30
    ) -> List[Tuple[float, float]]:
        """Curate top segments from sampled video content."""
        # Sample video segments
        sampled_segments = self.sample_video_segments()

        # Analyze segment importance
        segment_scores = [
            (segment, self.analyze_segment_importance(segment[0]))
            for segment in sampled_segments
        ]

        # Sort segments by importance score
        segment_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top segments for curation
        top_segments = segment_scores[:num_curated]

        # Prepare final curated segments
        curated_segments = [
            (segment[1], segment[2])  # Original start and end times
            for segment, _ in top_segments
        ]

        return curated_segments

    def create_curated_videos(
        self, num_curated: int = 4, target_duration: int = 30
    ) -> List[str]:
        """Create multiple curated video highlights."""
        # Get top segments
        top_segments = self.curate_top_segments(num_curated, target_duration)

        # Create curated videos
        curated_videos = []
        for i, (start, end) in enumerate(top_segments, 1):
            output_video = os.path.join(OUTPUT_DIR, f"curated_video_{i:02d}.mp4")

            # Crop video segment
            crop_video(self.input_video, output_video, [(start, end)])

            curated_videos.append(output_video)
            print(f"Created curated video {i}: {start:.2f}s - {end:.2f}s")

        return curated_videos


def main():
    parser = argparse.ArgumentParser(description="Advanced Video Content Curation")
    parser.add_argument(
        "--input", type=str, default=INPUT_VIDEO, help="Path to input video file"
    )
    parser.add_argument(
        "--num_curated",
        type=int,
        default=4,
        help="Number of curated video highlights to create",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=30,
        help="Duration of each curated video in seconds",
    )

    args = parser.parse_args()

    try:
        # Initialize video analyzer
        video_analyzer = AdvancedVideoAnalyzer(args.input)

        # Create curated videos
        curated_videos = video_analyzer.create_curated_videos(
            num_curated=args.num_curated, target_duration=args.target_duration
        )

        print("\nCurated videos created successfully:")
        for video in curated_videos:
            print(f"- {video}")

    except Exception as e:
        print(f"Error during video curation: {str(e)}")
        raise


if __name__ == "__main__":
    main()