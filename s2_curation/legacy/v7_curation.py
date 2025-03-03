import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import shlex
import subprocess
import spacy
import nltk
from collections import Counter, deque
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from textblob import TextBlob
import re
import argparse

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced paths with absolute paths
INPUT_VIDEO = os.path.abspath("../downloads/input_video.mp4")
AUDIO_FILE = os.path.abspath("../downloads/extracted_audio.wav")
OUTPUT_DIR = os.path.abspath("../downloads")
TEMP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "temp"))

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    """Extract audio with enhanced error handling and shlex for command handling."""
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


def find_best_segments(words, target_duration=45, min_segment_duration=5):
    """Find best segments with improved selection criteria and duration handling."""
    segments = []
    current_segment = []
    current_text = []

    # First pass: Create initial segments
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

            # Slide window with smaller overlap for more segments
            overlap = 3  # Reduced overlap for more segment options
            current_segment = current_segment[-overlap:]
            current_text = current_text[-overlap:]

    if not segments:
        raise ValueError("No valid segments found in the video")

    # Sort segments by score
    segments.sort(key=lambda x: x.score, reverse=True)

    # Filter out overlapping segments
    def segments_overlap(seg1, seg2, threshold=0.5):
        overlap_start = max(seg1.start, seg2.start)
        overlap_end = min(seg1.end, seg2.end)
        if overlap_end <= overlap_start:
            return False
        overlap_duration = overlap_end - overlap_start
        return overlap_duration > threshold * min(seg1.duration, seg2.duration)

    # Select best segments with duration target
    selected_segments = []
    total_duration = 0
    min_target = target_duration * 0.8  # Allow 20% below target

    # First pass: Add highest scoring segments
    for segment in segments:
        # Check if this segment overlaps with any selected segment
        if any(segments_overlap(segment, selected) for selected in selected_segments):
            continue

        if total_duration + segment.duration <= target_duration:
            selected_segments.append(segment)
            total_duration += segment.duration

    # If we're below minimum target, try to add more segments
    if total_duration < min_target:
        # Second pass: Try to fill gaps with smaller segments
        remaining_segments = [s for s in segments if s not in selected_segments]
        for segment in remaining_segments:
            if any(
                segments_overlap(segment, selected) for selected in selected_segments
            ):
                continue

            if total_duration + segment.duration <= target_duration:
                selected_segments.append(segment)
                total_duration += segment.duration

            if total_duration >= min_target:
                break

    # If still below target, adjust segment durations
    if total_duration < min_target and selected_segments:
        # Try to extend segments
        extension_per_segment = (min_target - total_duration) / len(selected_segments)
        for segment in selected_segments:
            # Find the original word list for this segment
            segment_words = [
                w for w in words if w.start >= segment.start and w.end <= segment.end
            ]
            if segment_words:
                # Try to extend the segment
                extension = min(2.0, extension_per_segment)  # Max 2 seconds extension
                # Find nearby words to extend the segment
                next_words = [
                    w
                    for w in words
                    if w.start >= segment.end and w.start <= segment.end + extension
                ]
                if next_words:
                    segment.end = next_words[-1].end
                    segment.duration = segment.end - segment.start

    # Sort selected segments by start time
    selected_segments.sort(key=lambda x: x.start)

    # Print warning if we're significantly under target
    final_duration = sum(segment.duration for segment in selected_segments)
    if final_duration < min_target:
        print(
            f"\nWarning: Could only find {final_duration:.2f} seconds of good content."
        )
        print("Consider adjusting the minimum segment duration or engagement criteria.")

    return [(segment.start, segment.end) for segment in selected_segments]


def calculate_engagement_score(text):
    """Calculate an engagement score with adjusted weights for better segment selection."""
    # Sentiment analysis
    sentiment_score = analyzer.polarity_scores(text)["compound"]

    # Question detection (increased weight)
    question_bonus = 3 if "?" in text else 0

    # Emotion words detection
    blob = TextBlob(text)
    emotion_words = sum(1 for word, tag in blob.tags if tag.startswith("JJ"))

    # Key phrase detection (expanded list and increased weight)
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

    # Add bonus for longer segments that make sense
    length_bonus = len(text.split()) / 10  # Bonus for longer coherent segments

    # Calculate final score with adjusted weights
    engagement_score = (
        abs(sentiment_score) * 3  # Increased weight for sentiment
        + question_bonus
        + emotion_words * 1.5  # Increased weight for emotional content
        + phrase_bonus
        + length_bonus
    )

    return engagement_score


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


def get_output_path(duration):
    """Generate output path based on selected duration."""
    return os.path.abspath(os.path.join(OUTPUT_DIR, f"curated_video_{duration}s.mp4"))


import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import shlex
import subprocess
import spacy
import nltk
from collections import Counter, deque
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from textblob import TextBlob
import re
import argparse

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced paths with absolute paths
INPUT_VIDEO = os.path.abspath("../downloads/input_video.mp4")
OUTPUT_DIR = os.path.abspath("../downloads")
TEMP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "temp"))

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize existing helper classes and functions from the previous script
# (Include VideoSegment, calculate_engagement_score, etc. from the previous script)


def split_video_into_segments(input_video, segment_duration=60):
    """Split video into fixed-duration segments."""
    temp_segments_dir = os.path.join(TEMP_DIR, "segments")
    os.makedirs(temp_segments_dir, exist_ok=True)

    segments = []

    # Get total video duration
    probe_cmd = shlex.split(
        f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_video}"'
    )
    total_duration = float(subprocess.check_output(probe_cmd).decode().strip())

    # Split video into segments
    for start in np.arange(0, total_duration, segment_duration):
        end = min(start + segment_duration, total_duration)
        segment_file = os.path.join(
            temp_segments_dir, f"segment_{int(start//segment_duration):02d}.mp4"
        )

        split_cmd = shlex.split(
            f'ffmpeg -i "{input_video}" '
            f"-ss {start} "
            f"-t {end - start} "
            f"-c:v libx264 -preset fast "
            f"-c:a aac "
            f'"{segment_file}"'
        )

        subprocess.run(split_cmd, check=True, capture_output=True)
        segments.append((segment_file, start, end))

    return segments


def process_video_segments(segments, target_duration):
    """Process video segments and select best content."""
    final_segments = []
    total_curated_duration = 0

    for segment_file, segment_start, segment_end in segments:
        try:
            # Extract audio for this segment
            segment_audio = os.path.join(
                TEMP_DIR, f"segment_audio_{os.path.basename(segment_file)}.wav"
            )
            extract_audio(segment_file, segment_audio)

            # Transcribe segment audio
            words = transcribe_audio(segment_audio)

            # Find best segments within this segment
            best_segment_times = find_best_segments(
                words,
                target_duration=min(segment_end - segment_start, target_duration),
                min_segment_duration=5,
            )

            # Adjust segment times relative to original video
            adjusted_segments = [
                (start + segment_start, end + segment_start)
                for start, end in best_segment_times
            ]

            final_segments.extend(adjusted_segments)

            # Stop if we've reached target duration
            total_curated_duration = sum(end - start for start, end in final_segments)
            if total_curated_duration >= target_duration:
                break

        except Exception as e:
            print(f"Error processing segment {segment_file}: {e}")
        finally:
            # Clean up temporary audio file
            if os.path.exists(segment_audio):
                os.remove(segment_audio)

    # Trim or adjust segments if needed
    final_segments = sorted(final_segments, key=lambda x: x[0])

    # If no good segments found, return first segment as fallback
    if not final_segments:
        first_segment = segments[0]
        return [(first_segment[1], first_segment[2])]

    # Select segments up to target duration
    selected_segments = []
    current_duration = 0
    for start, end in final_segments:
        segment_duration = end - start
        if current_duration + segment_duration <= target_duration:
            selected_segments.append((start, end))
            current_duration += segment_duration
        else:
            break

    return selected_segments


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create curated video with specified duration from large input"
    )
    parser.add_argument(
        "--duration",
        type=int,
        choices=[30, 45, 60],
        default=45,
        help="Target duration in seconds (30, 45, or 60)",
    )

    args = parser.parse_args()
    target_duration = args.duration

    output_video = get_output_path(target_duration)

    try:
        # Split video into segments
        print("Splitting video into processing segments...")
        video_segments = split_video_into_segments(INPUT_VIDEO)

        # Process segments and find best content
        print(f"Finding best content for {target_duration}s video...")
        best_segments = process_video_segments(video_segments, target_duration)

        # Crop and save video
        print("Creating final curated video...")
        crop_video(INPUT_VIDEO, output_video, best_segments)

        print(f"Curated video successfully saved to: {output_video}")

        # Print segment information
        print("\nSelected segments:")
        total_duration = 0
        for i, (start, end) in enumerate(best_segments, 1):
            duration = end - start
            total_duration += duration
            print(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {duration:.2f}s)")
        print(f"\nTotal Duration: {total_duration:.2f}s")

    except Exception as e:
        print(f"Error during video curation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
