import os
from openai import OpenAI
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from dotenv import load_dotenv

# Configuration
FONT_SIZE = 40
TEXT_COLOR = "white"
STROKE_COLOR = "black"
STROKE_WIDTH = 2


def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise


def generate_transcript(audio_path, api_key):
    """Generate transcript using OpenAI Whisper"""
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
        raise


def group_words_into_phrases(words, max_words=3):
    """Group words into small phrases for better readability"""
    phrases = []
    current_phrase = []
    current_count = 0

    for word in words:
        current_phrase.append(word)
        current_count += 1

        if current_count >= max_words:
            if current_phrase:
                start_time = current_phrase[0].start
                end_time = current_phrase[-1].end
                text = " ".join(w.word for w in current_phrase)
                phrases.append({"text": text, "start": start_time, "end": end_time})
                current_phrase = []
                current_count = 0

    # Add remaining words
    if current_phrase:
        start_time = current_phrase[0].start
        end_time = current_phrase[-1].end
        text = " ".join(w.word for w in current_phrase)
        phrases.append({"text": text, "start": start_time, "end": end_time})

    return phrases


def create_subtitle_clips(phrases, video_size):
    """Create subtitle clips with TikTok-style animation"""
    clips = []

    for phrase in phrases:
        # Create text clip
        clip = TextClip(
            phrase["text"],
            # fontsize=FONT_SIZE,
            color=TEXT_COLOR,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method="caption",
            align="center",
            size=(video_size[0] * 0.8, None),  # Wrap text if needed
        )

        # Position at bottom center
        position = ("center", video_size[1] * 0.8)

        # Add timing and position
        clip = clip.set_position(position)
        clip = clip.set_start(phrase["start"])
        clip = clip.set_duration(phrase["end"] - phrase["start"])

        # Add fade effects
        clip = clip.crossfadein(0.2).crossfadeout(0.2)

        clips.append(clip)

    return clips


def create_subtitled_video(video_path, transcript, output_path):
    """Create final video with TikTok-style subtitles"""
    try:
        # Load video
        video = VideoFileClip(video_path)

        # Group words into phrases
        phrases = group_words_into_phrases(transcript.words)

        # Create subtitle clips
        subtitle_clips = create_subtitle_clips(phrases, video.size)

        # Combine video with subtitles
        final_video = CompositeVideoClip([video] + subtitle_clips)

        # Write final video
        final_video.write_videofile(
            output_path, fps=video.fps, codec="libx264", audio_codec="aac"
        )

        # Clean up
        video.close()
        final_video.close()

    except Exception as e:
        print(f"Error creating subtitled video: {e}")
        raise


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Define paths
        video_path = "../downloads/reframed_video_01.mp4"
        audio_path = "../downloads/temp_audio.wav"
        output_path = "../downloads/final_video.mp4"

        # Ensure input video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Extract audio
        print("Extracting audio...")
        extract_audio(video_path, audio_path)

        # Generate transcript
        print("Generating transcript...")
        transcript = generate_transcript(audio_path, os.getenv("OPENAI_API_KEY"))

        # Create subtitled video
        print("Creating subtitled video...")
        create_subtitled_video(video_path, transcript, output_path)

        # Clean up temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"Successfully created video: {output_path}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()