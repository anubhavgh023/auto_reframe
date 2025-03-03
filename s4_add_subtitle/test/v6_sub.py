import os
from dotenv import load_dotenv
from openai import OpenAI
from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
    AudioFileClip,
)
import numpy as np

# Global Configuration
FONT_SIZE_RATIO = 0.045  # Relative to frame height
HIGHLIGHT_COLOR = "yellow"
BOX_COLOR = (101, 13, 168)  # RGB for purple background
STROKE_COLOR = "black"
STROKE_WIDTH = 1.5
HIGHLIGHT_MODE = "per_word"  # Options: "per_word" or "box"
WORD_SPACING = 10  # Spacing between words in pixels
BASE_TEXT_COLOR = "white"
FONT = "Arial"  # Default to Arial, can be changed to custom font path


def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()


def generate_transcript(audio_path, api_key):
    """Generate word-level timestamps using OpenAI Whisper"""
    client = OpenAI(api_key=api_key)
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        return transcript
    except Exception as e:
        print(f"Error generating transcript: {e}")
        return None


def create_scrolling_word_clip(word, frame_size, word_timing, total_duration):
    """Create a scrolling word clip with highlighting"""
    fontsize = int(frame_size[1] * FONT_SIZE_RATIO)

    # Create base (white) word clip
    base_clip = TextClip(
        word.word,
        font=FONT,
        fontsize=fontsize,
        color=BASE_TEXT_COLOR,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
        method="label",
    )

    # Calculate the scrolling animation
    def scroll_position(t):
        # Start position (off-screen right)
        start_x = frame_size[0] + 50
        # End position (off-screen left)
        end_x = -base_clip.w - 50

        # Calculate progress through the word's duration
        word_start = float(word.start)
        word_end = float(word.end)
        word_duration = word_end - word_start

        # Smooth scrolling with easing
        if t < word_start:
            return (start_x, frame_size[1] * 0.7)
        elif t > word_end:
            return (end_x, frame_size[1] * 0.7)
        else:
            progress = (t - word_start) / word_duration
            x = start_x + (end_x - start_x) * progress
            return (x, frame_size[1] * 0.7)

    # Apply scrolling animation
    base_clip = base_clip.set_position(scroll_position)
    base_clip = base_clip.set_start(0)
    base_clip = base_clip.set_duration(total_duration)

    clips = [base_clip]

    # Add highlighted version during word's active time
    if HIGHLIGHT_MODE == "per_word":
        highlight_clip = TextClip(
            word.word,
            font=FONT,
            fontsize=fontsize,
            color=HIGHLIGHT_COLOR,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method="label",
        )
        highlight_clip = highlight_clip.set_position(scroll_position)
        highlight_clip = highlight_clip.set_start(float(word.start))
        highlight_clip = highlight_clip.set_duration(float(word.end - word.start))
        highlight_clip = highlight_clip.crossfadein(0.1).crossfadeout(0.1)
        clips.append(highlight_clip)

    return clips


def create_final_video(video_path, transcript, output_path):
    """Create final video with scrolling subtitles"""
    try:
        # Load video
        video = VideoFileClip(video_path)
        frame_size = video.size

        # Create clips for each word
        all_clips = []
        for word in transcript.words:
            word_clips = create_scrolling_word_clip(
                word, frame_size, (float(word.start), float(word.end)), video.duration
            )
            all_clips.extend(word_clips)

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

    # Define paths
    video_path = "../downloads/reframed_video_01.mp4"
    audio_path = "../downloads/temp_audio.wav"
    output_path = "../downloads/final_video.mp4"

    # Extract audio
    print("Extracting audio...")
    extract_audio(video_path, audio_path)

    # Generate transcript
    print("Generating transcript...")
    transcript = generate_transcript(audio_path, os.getenv("OPENAI_API_KEY"))
    if not transcript:
        return

    # Create final video with scrolling subtitles
    print("Creating final video with scrolling subtitles...")
    final_video = create_final_video(video_path, transcript, output_path)

    # Clean up temporary files
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if final_video:
        print(f"Successfully created video: {output_path}")


if __name__ == "__main__":
    main()