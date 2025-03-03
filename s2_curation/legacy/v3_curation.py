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

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
INPUT_VIDEO = "../downloads/input_video.mp4"
AUDIO_FILE = "../downloads/extracted_audio.wav"
OUTPUT_DIR = "../downloads"
CURATED_VIDEO = os.path.join(OUTPUT_DIR, "curated_video.mp4")

# Download NLTK data if not already downloaded
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    print("Downloading vader_lexicon...")
    nltk.download("vader_lexicon")

# Initialize spaCy and NLTK Sentiment Analyzer
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()


def extract_audio(input_video, output_audio):
    """Extract audio from video using ffmpeg, using shlex for command construction."""
    try:
        cmd = f'ffmpeg -i "{input_video}" -q:a 0 -map a "{output_audio}"'
        cmd_list = shlex.split(cmd)  # Split the command string into a list

        subprocess.run(
            cmd_list, check=True, capture_output=True, text=True
        )  # changed the cmd to cmd_list
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction error: {e.stderr}")
        raise


def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper"""
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
        print(f"Transcription error: {e}")
        raise


def extract_keywords(text, num_keywords=10):
    """Extract keywords from text using spaCy."""
    doc = nlp(text)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "ADJ"]
    ]
    keyword_counts = Counter(keywords)
    return [word for word, count in keyword_counts.most_common(num_keywords)]


def get_sentiment_score(text):
    """Get the sentiment score of a text using NLTK VADER."""
    scores = analyzer.polarity_scores(text)
    return scores["compound"]  # Compound score


def find_best_segments(words, target_duration=45):
    """Find best segments using a combination of keyword extraction, sentiment analysis, and GPT-4 (optional)."""
    transcript_text = "\n".join(
        [f"{word.start:.2f}-{word.end:.2f}: {word.word}" for word in words]
    )
    all_segments = []
    segment_size = 10  # seconds

    for i in range(0, len(words), segment_size):
        start_time = words[i].start
        end_time = words[min(i + segment_size - 1, len(words) - 1)].end
        segment_text = " ".join(
            [word.word for word in words[i : min(i + segment_size, len(words))]]
        )
        keywords = extract_keywords(segment_text)
        sentiment_score = get_sentiment_score(segment_text)
        segment_score = len(keywords) + abs(sentiment_score)  # combine the score

        all_segments.append(
            {
                "start": start_time,
                "end": end_time,
                "text": segment_text,
                "score": segment_score,
            }
        )

    sorted_segments = sorted(all_segments, key=lambda x: x["score"], reverse=True)

    best_segments = []
    current_duration = 0
    for segment in sorted_segments:
        if current_duration + (segment["end"] - segment["start"]) <= target_duration:
            best_segments.append((segment["start"], segment["end"]))
            current_duration += segment["end"] - segment["start"]

    return best_segments


def crop_video(input_video, output_video, segments):
    """Crop video using ffmpeg"""
    try:
        # Prepare filter complex command for precise segment extraction
        filter_parts = []
        input_parts = []

        for i, (start, end) in enumerate(segments):
            filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
            )
            input_parts.append(f"[v{i}][a{i}]")

        # Concatenate filter
        filter_parts.append(
            f"{''.join(input_parts)}concat=n={len(segments)}:v=1:a=1[outv][outa]"
        )

        # Full FFmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            input_video,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_video,
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_video

    except subprocess.CalledProcessError as e:
        print(f"Video cropping error: {e.stderr}")
        raise


def main():
    # Extract audio
    audio_path = extract_audio(INPUT_VIDEO, AUDIO_FILE)

    # Transcribe audio
    words = transcribe_audio(AUDIO_FILE)
    print(words)
    print("\n-------------\n")

    # Find best segment (using first target duration)
    best_segments = find_best_segments(words, 45)
    print(best_segments)

    # Crop and save video
    crop_video(INPUT_VIDEO, CURATED_VIDEO, best_segments)

    print(f"Curated video saved to: {CURATED_VIDEO}")


if __name__ == "__main__":
    main()
