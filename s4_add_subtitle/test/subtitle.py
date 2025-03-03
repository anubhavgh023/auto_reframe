import os
import ffmpeg
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

from moviepy import (
    TextClip,
    CompositeVideoClip,
    VideoFileClip,
    ColorClip,
    AudioFileClip,
)

# Subtitle Styling Configuration
FONT = "helpers/fonts/Chewy-Regular.ttf"
FONT_SIZE_RATIO = 0.045  # Relative to frame height
HIGHLIGHT_COLOR = "yellow"
BOX_COLOR = (101, 13, 168)  # RGB for blue
STROKE_COLOR = "black"
STROKE_WIDTH = 1.5
HIGHLIGHT_MODE = "box"  # Options: "per_word" or "box"
WORD_SPACING = 10  # Spacing between words in pixels
BASE_TEXT_COLOR = "white"


def extract_audio(input_video_path, output_audio_path):
    """
    Extract audio from input video using ffmpeg
    """
    try:
        # Extract audio from video
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_video_path,
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",  # Audio codec
            "-ar",
            "16000",  # Audio sample rate (16kHz for Whisper)
            "-ac",
            "1",  # Mono audio
            output_audio_path,
        ]

        subprocess.run(ffmpeg_command, check=True)
        print(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None


def generate_subtitles(audio_file_path, api_key):
    """Generate word-level timestamps using OpenAI Whisper"""
    client = OpenAI(api_key=api_key)
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        return transcript
    except Exception as e:
        print(f"Error generating subtitles: {e}")
        return None


def group_words_into_sentences(words, max_words=3):
    """Group words into sentences based on timing and max words"""
    sentences = []
    current_sentence = []
    word_count = 0
    last_end_time = 0

    for word in words:
        # Start new sentence if max words reached or large time gap (> 0.7s)
        if word_count >= max_words or (
            word.start - last_end_time > 0.7 and word_count > 0
        ):
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
                word_count = 0

        current_sentence.append(word)
        word_count += 1
        last_end_time = word.end

    if current_sentence:  # Add remaining words
        sentences.append(current_sentence)

    return sentences


# ---
def create_box_clip(word, frame_size, position, duration):
    # Modify TextClip creation
    temp_clip = TextClip(
        word.word,
        font=FONT,
        fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
        color=BASE_TEXT_COLOR,
    )


def create_sentence_clips(sentence_words, frame_size):
    # Modify word clip creations
    clip = TextClip(
        word.word,
        font=FONT,
        fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
        color=BASE_TEXT_COLOR,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
    )

    # Similar modification for highlight clip
    highlight_clip = TextClip(
        word.word,
        font=FONT,
        fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
        color=HIGHLIGHT_COLOR,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
    )


# ---



def add_subtitles_to_video(video_path, transcript, output_path):
    """Create final video with TikTok-style subtitles"""
    try:
        # Load video
        video = VideoFileClip(video_path)
        frame_size = video.size

        # Group words into sentences
        sentences = group_words_into_sentences(transcript.words)

        # Create clips for each sentence with word highlighting
        all_clips = []
        for sentence in sentences:
            sentence_clips = create_sentence_clips(sentence, frame_size)
            all_clips.extend(sentence_clips)

        # Combine all clips
        final_video = CompositeVideoClip([video] + all_clips, size=frame_size)

        # Write final video
        final_video.write_videofile(
            output_path, fps=24, codec="libx264", audio_codec="aac"
        )

        # Clean up
        video.close()
        final_video.close()

        return output_path

    except Exception as e:
        print(f"Error creating final video: {e}")
        return None


def main():
    # Load environment variables
    load_dotenv()

    # Input and output paths
    input_video_path = "../downloads/reframed_video_01.mp4"
    temp_audio_path = "../downloads/extracted_audio.wav"
    final_video_path = "../downloads/subtitled_video.mp4"

    # Step 1: Extract audio from video
    audio_path = extract_audio(input_video_path, temp_audio_path)
    if not audio_path:
        print("Failed to extract audio")
        return

    # Step 2: Generate subtitles using OpenAI Whisper
    transcript = generate_subtitles(audio_path, os.getenv("OPENAI_API_KEY"))
    if not transcript:
        print("Failed to generate subtitles")
        return

    # Step 3: Add subtitles to video
    final_video = add_subtitles_to_video(input_video_path, transcript, final_video_path)

    # Optional: Clean up temporary audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    if final_video:
        print(f"Successfully created subtitled video: {final_video_path}")


if __name__ == "__main__":
    main()
