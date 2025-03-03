import os
import json
import random
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


def get_video_duration(input_video: str) -> float:
    """Get total video duration in seconds."""
    probe_cmd = shlex.split(
        f"ffprobe -v error -show_entries format=duration "
        f'-of default=noprint_wrappers=1:nokey=1 "{input_video}"'
    )
    return float(subprocess.check_output(probe_cmd).decode().strip())


def split_video_into_chunks(input_video: str, chunk_duration: int = 60) -> List[str]:
    """Split video into 1-minute chunks."""
    total_duration = get_video_duration(input_video)
    chunks = []

    # Create temp directory for chunks
    chunks_dir = os.path.join(TEMP_DIR, "video_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, int(total_duration))
        chunk_file = os.path.join(chunks_dir, f"chunk_{start}_{end}.mp4")

        # Extract video chunk
        cmd = shlex.split(
            f'ffmpeg -i "{input_video}" '
            f"-ss {start} "
            f"-t {end - start} "
            f"-c:v libx264 -preset fast "
            f"-c:a aac "
            f'"{chunk_file}"'
        )
        subprocess.run(cmd, check=True, capture_output=True)
        chunks.append(chunk_file)

    return chunks


def extract_audio_from_chunks(video_chunks: List[str]) -> List[str]:
    """Extract audio from video chunks."""
    audio_files = []
    
    for chunk in video_chunks:
        audio_file = chunk.replace(".mp4", ".wav")
        cmd = shlex.split(
            f'ffmpeg -i "{chunk}" '
            f"-vn -acodec pcm_s16le "
            f"-ar 44100 -ac 2 "
            f'-y "{audio_file}"'
        )
        subprocess.run(cmd, check=True, capture_output=True)
        audio_files.append(audio_file)
    
    return audio_files


def transcribe_audio_chunks(audio_files: List[str]) -> List[Dict[str, Any]]:
    """Transcribe audio chunks using OpenAI's Whisper."""
    transcriptions = []
    
    for audio_path in audio_files:
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            
            # Save transcription to JSON
            transcription_file = audio_path.replace(".wav", "_transcript.json")
            with open(transcription_file, "w") as f:
                json.dump(transcript.model_dump(), f)
            
            transcriptions.append({
                "audio_path": audio_path,
                "transcript_path": transcription_file,
                "text": " ".join(word.word for word in transcript.words)
            })
        
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
    
    return transcriptions


def calculate_transcription_importance(transcription: Dict[str, Any]) -> float:
    """Calculate the importance of a transcription using multiple metrics."""
    text = transcription["text"]
    
    # Sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = abs(sentiment_analyzer.polarity_scores(text)["compound"])

    # Spacy entity recognition
    doc = nlp(text)
    named_entities = len(doc.ents)

    # Key phrase detection
    key_phrases = [
        "how to", "why", "best", "top", "secret", "important", 
        "fascinating", "learn", "discover", "explain"
    ]
    phrase_bonus = sum(4 for phrase in key_phrases if phrase in text.lower())

    # Question and key action word detection
    question_bonus = 3 if "?" in text else 0
    
    # TextBlob for emotion and descriptive words
    blob = TextBlob(text)
    emotion_words = sum(1 for word, tag in blob.tags if tag.startswith("JJ"))

    # Comprehensive importance score
    importance_score = (
        sentiment_score * 3
        + named_entities * 2
        + phrase_bonus
        + question_bonus
        + emotion_words * 1.5
    )

    return importance_score


def select_top_transcriptions(
    transcriptions: List[Dict[str, Any]], 
    max_curated: int = 4
) -> List[Dict[str, Any]]:
    """Select top transcriptions based on importance score."""
    # Calculate importance for each transcription
    scored_transcriptions = [
        {
            **transcription, 
            "importance_score": calculate_transcription_importance(transcription)
        }
        for transcription in transcriptions
    ]
    
    # Sort by importance score in descending order
    sorted_transcriptions = sorted(
        scored_transcriptions, 
        key=lambda x: x["importance_score"], 
        reverse=True
    )
    
    # Select top transcriptions, minimum of 2
    num_curated = min(max_curated, len(sorted_transcriptions), max(2, max_curated//2))
    return sorted_transcriptions[:num_curated]


def create_curated_videos(
    input_video: str, 
    selected_transcriptions: List[Dict[str, Any]]
) -> List[str]:
    """Create curated videos with 1.25x speed."""
    curated_videos = []
    
    for i, transcription in enumerate(selected_transcriptions, 1):
        # Extract original chunk details
        audio_path = transcription["audio_path"]
        original_chunk = audio_path.replace(".wav", ".mp4")
        
        # Output curated video path
        output_video = os.path.join(OUTPUT_DIR, f"curated_video_{i:02d}.mp4")
        
        # Use ffmpeg to create curated video with 1.25x speed

        cmd = shlex.split(
        f'ffmpeg -i "{original_chunk}" '
        f'-filter_complex "setpts=0.8*PTS,atempo=1.25" '  # Apply both video and audio filters
        f'"{output_video}"'
        )

        
        subprocess.run(cmd, check=True, capture_output=True)
        curated_videos.append(output_video)
        
        print(f"Created curated video {i}: {output_video}")
    
    return curated_videos


def main():
    parser = argparse.ArgumentParser(description="Advanced Video Content Curation")
    parser.add_argument(
        "--input", type=str, default=INPUT_VIDEO, help="Path to input video file"
    )
    parser.add_argument(
        "--max_curated",
        type=int,
        default=4,
        help="Maximum number of curated video highlights to create",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=60,
        help="Duration of video chunks in seconds",
    )

    args = parser.parse_args()

    try:
        # Split video into chunks
        video_chunks = split_video_into_chunks(
            args.input, 
            chunk_duration=args.chunk_duration
        )
        
        # Extract audio from chunks
        audio_files = extract_audio_from_chunks(video_chunks)
        
        # Transcribe audio chunks
        transcriptions = transcribe_audio_chunks(audio_files)
        
        # Select top transcriptions
        top_transcriptions = select_top_transcriptions(
            transcriptions, 
            max_curated=args.max_curated
        )
        
        # Create curated videos
        curated_videos = create_curated_videos(
            args.input, 
            top_transcriptions
        )

        print("\nCurated videos created successfully:")
        for video in curated_videos:
            print(f"- {video}")

    except Exception as e:
        print(f"Error during video curation: {str(e)}")
        raise


if __name__ == "__main__":
    main()