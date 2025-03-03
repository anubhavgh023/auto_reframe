import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from openai import OpenAI
from dotenv import load_dotenv

FONT = "4.add_subtitle/fonts/Chewy-Regular.ttf"  # Using a standard font for better compatibility
FONT_SIZE_RATIO = 0.045
BOX_COLOR = (101, 13, 168)
STROKE_COLOR = "black"
STROKE_WIDTH = 1.5
WORD_SPACING = 10
BASE_TEXT_COLOR = "white"


def generate_subtitles(audio_file_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    return transcript


def group_words_into_sentences(words, max_words=3):
    sentences = []
    current_sentence = []
    word_count = 0
    last_end_time = 0

    for word in words:
        if word_count >= max_words or (
            word["start"] - last_end_time > 0.7 and word_count > 0
        ):
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
                word_count = 0

        current_sentence.append(word)
        word_count += 1
        last_end_time = word["end"]

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def create_box_clip(word, frame_size, position, duration):
    padding = 9
    temp_clip = TextClip(
        word["text"],
        font=FONT,
        fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
        color=BASE_TEXT_COLOR,
        method="label",
    )

    box = ColorClip(
        size=(int(temp_clip.w + 1.2 * padding), int(temp_clip.h + padding)),
        color=BOX_COLOR,
    )

    box = box.set_position((position[0] - padding / 2, position[1] - padding / 4))
    box = box.set_duration(duration)

    temp_clip.close()
    return box


def create_sentence_clips(sentence_words, frame_size):
    clips = []
    base_y = frame_size[1] * 0.7
    total_width = 0
    word_clips = []

    for word in sentence_words:
        clip = TextClip(
            word["text"],
            font=FONT,
            fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
            color=BASE_TEXT_COLOR,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method="label",
        )
        word_clips.append((word, clip))
        total_width += clip.w + WORD_SPACING

    start_x = (frame_size[0] - total_width) / 2
    current_x = start_x

    for word, clip in word_clips:
        duration = float(word["end"] - word["start"])
        box_clip = create_box_clip(word, frame_size, (current_x, base_y), duration)
        box_clip = box_clip.set_start(float(word["start"]))
        clips.append(box_clip)

        white_clip = clip.set_position((current_x, base_y))
        white_clip = white_clip.set_start(sentence_words[0]["start"])
        white_clip = white_clip.set_duration(
            sentence_words[-1]["end"] - sentence_words[0]["start"]
        )
        clips.append(white_clip)

        current_x += clip.w + WORD_SPACING

    clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in clips]
    return clips


def process_video(video_path, output_path):
    video = VideoFileClip(video_path)

    # Extract audio
    audio_path = "../downloads/extracted_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Generate subtitles
    print("Generating subtitles from audio...")
    transcript = generate_subtitles(audio_path, os.getenv("OPENAI_API_KEY"))

    # Create subtitled video
    print("Creating final video with subtitles...")
    frame_size = video.size
    sentences = group_words_into_sentences(transcript["words"])

    all_clips = []
    for sentence in sentences:
        sentence_clips = create_sentence_clips(sentence, frame_size)
        all_clips.extend(sentence_clips)

    final_video = CompositeVideoClip([video] + all_clips)
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

    # Cleanup
    video.close()
    final_video.close()
    os.remove(audio_path)

    return output_path


def main():
    load_dotenv()
    video_path = "../downloads/reframed_video_01.mp4"
    output_path = "../downloads/final_video.mp4"

    try:
        process_video(video_path, output_path)
        print(f"Successfully created video: {output_path}")
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
