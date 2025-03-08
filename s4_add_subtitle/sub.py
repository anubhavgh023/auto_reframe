import os
import json
from dotenv import load_dotenv
import concurrent.futures
from moviepy.editor import (
    TextClip,
    CompositeVideoClip,
    VideoFileClip,
    ColorClip,
    AudioFileClip,
)
import moviepy.video.fx.all
import tempfile
import time
from functools import lru_cache
import random
from openai import OpenAI
import subprocess
import shlex

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input and output paths (output_folder and temp_folder remain constant)
output_folder = os.path.join(script_dir, "../downloads/final_videos")
temp_folder = os.path.join(script_dir, "temp/")

# Subtitle Styles Configuration with integrated fonts
SUBTITLE_STYLES = {
    "style1": {
        "fontStyle": "OpenSans-ExtraBold.ttf",
        "color": "white",
        "highlight-mode": "per_word",
        "font-size": 0.05,
        "stroke-width": 1.7,
        "highlight-color": "yellow",
        "stroke-color": "black",
    },
    "style2": {
        "fontStyle": "Alfa-Slab.ttf",
        "color": "white",
        "highlight-mode": "per_word",
        "font-size": 0.045,
        "stroke-width": 1.5,
        "highlight-color": "#04ff00",  # bright green
        "stroke-color": "black",
    },
    "style3": {
        "fontStyle": "Chewy-Regular.ttf",
        "color": "white",
        "highlight-mode": "per_word",
        "font-size": 0.055,
        "stroke-width": 1.5,
        "highlight-color": "red",
        "stroke-color": "black",
    },
}

# Global Configuration
BOX_COLOR = (20, 40, 120)
WORD_SPACING = 8
MAX_LINE_WIDTH_RATIO = 0.8  # 80% of frame width
MAX_WORDS_PER_SENTENCE = 3

# Set MoviePy's global parameters
import moviepy.config as mpconfig
mpconfig.FFMPEG_BINARY = "ffmpeg"

@lru_cache(maxsize=512)
def get_text_dimensions(text, font_size, font_file):
    """Get text dimensions without creating full clip (cached)"""
    temp_clip = TextClip(
        text,
        font=font_file,
        fontsize=font_size,
        color="white",
        stroke_color="black",
        stroke_width=1,
        method="label",
    )
    dimensions = (temp_clip.w, temp_clip.h)
    temp_clip.close()
    return dimensions

def extract_audio(video_path, output_audio_path):
    """Extract audio from video using ffmpeg with shlex and subprocess"""
    try:
        cmd = f"ffmpeg -i {video_path} -acodec mp3 -vn -y {output_audio_path}"
        cmd_args = shlex.split(cmd)
        result = subprocess.run(cmd_args, capture_output=True, text=True, check=True)
        print(f"Audio extracted successfully: {result.stdout}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error extracting audio: {e}")
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

class WordInfo:
    """Class to store word information"""
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

def adjust_word_timestamps(transcript):
    """Adjust word timestamps from TranscriptionWord objects (assuming video starts at 0.0)"""
    words = []
    if not hasattr(transcript, "words") or not transcript.words:
        print("Invalid transcript data: missing 'words'")
        return words
    
    for word_data in transcript.words:
        start = word_data.start
        end = word_data.end
        if end > 0:
            start = max(0, start)
            if end <= start:
                end = start + 0.1
            words.append(WordInfo(word_data.word, start, end))
            # print(f"Word: {word_data.word}, Start: {start}, End: {end}")
    return words

def group_words_into_sentences(
    words, max_words=MAX_WORDS_PER_SENTENCE, frame_width=None, font_size=None, font_file=None
):
    """Group words into sentences with width, word count limits and proper sentence formation"""
    if not words:
        print("No words to group into sentences")
        return []
    
    sentences = []
    current_sentence = []
    word_count = 0
    last_end_time = 0
    current_width = 0
    max_width = frame_width * MAX_LINE_WIDTH_RATIO if frame_width else float('inf')

    for i, word in enumerate(words):
        word_width, _ = get_text_dimensions(word.word, font_size, font_file) if font_size and font_file else (0, 0)
        
        potential_width = current_width + word_width + (WORD_SPACING if current_sentence else 0)
        
        new_sentence_needed = (
            word_count >= max_words or
            (word.start - last_end_time > 0.7 and word_count > 0) or
            (potential_width > max_width and word_count > 0)
        )
        
        if new_sentence_needed:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
                word_count = 0
                current_width = 0
        
        current_sentence.append(word)
        word_count += 1
        last_end_time = word.end
        current_width = current_width + word_width + (WORD_SPACING if word_count > 1 else 0)
        
        words_to_add = min(max_words - word_count, len(words) - i - 1)
        if words_to_add > 0 and last_end_time + 0.7 >= words[i + 1].start:
            can_fit_more = True
            next_words_width = 0
            
            for j in range(1, words_to_add + 1):
                if i + j < len(words):
                    next_word = words[i + j]
                    next_word_width, _ = get_text_dimensions(next_word.word, font_size, font_file) if font_size and font_file else (0, 0)
                    if current_width + next_words_width + next_word_width + WORD_SPACING > max_width:
                        can_fit_more = False
                        break
                    next_words_width += next_word_width + WORD_SPACING
            
            if not can_fit_more or i == len(words) - 1:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                    word_count = 0
                    current_width = 0

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def create_box_clip(word, frame_size, position, duration, style):
    """Create a box background for a word"""
    padding = 9
    font_size = int(frame_size[1] * style["font-size"])
    word_w, word_h = get_text_dimensions(word.word, font_size, font_file)
    
    box = ColorClip(
        size=(int(word_w + 1.2 * padding), int(word_h + padding)),
        color=BOX_COLOR,
        ismask=False,
    )
    box = box.set_position((position[0] - padding / 2, position[1] - padding / 4))
    box = box.set_duration(duration)
    box = box.set_opacity(1)
    return box

def create_sentence_clips(sentence_words, frame_size, font_file, style_key="style1"):
    """Create clips for a sentence with per-word rendering for perfect alignment"""
    clips = []
    if not sentence_words:
        print("No sentence words provided, skipping clip creation")
        return clips
    
    style = SUBTITLE_STYLES.get(style_key, SUBTITLE_STYLES["style1"])
    font_size = int(frame_size[1] * style["font-size"])
    base_y = frame_size[1] * 0.7
    total_width = 0
    word_positions = []

    # Calculate total width and positions for all words
    for word in sentence_words:
        word_width, _ = get_text_dimensions(word.word, font_size, font_file)
        word_positions.append((word, word_width))
        total_width += word_width + WORD_SPACING

    # Center the sentence horizontally
    start_x = (frame_size[0] - total_width) / 2
    current_x = start_x

    # Create clips for each word
    for word, word_width in word_positions:
        # Base text clip (white/default color)
        base_clip = TextClip(
            word.word,
            font=font_file,
            fontsize=font_size,
            color=style["color"],
            stroke_color=style["stroke-color"],
            stroke_width=style["stroke-width"],
            method="label",
        ).set_position((current_x, base_y))
        base_clip = base_clip.set_start(sentence_words[0].start)
        base_clip = base_clip.set_duration(sentence_words[-1].end - sentence_words[0].start)
        base_clip = base_clip.crossfadein(0.1).crossfadeout(0.1)
        clips.append(base_clip)

        # Highlight clip (if applicable)
        if style["highlight-mode"] == "per_word":
            highlight_clip = TextClip(
                word.word,
                font=font_file,
                fontsize=font_size,
                color=style["highlight-color"],
                stroke_color=style["stroke-color"],
                stroke_width=style["stroke-width"],
                method="label",
            ).set_position((current_x, base_y))
            word_duration = word.end - word.start
            highlight_clip = highlight_clip.set_start(word.start).set_duration(word_duration)
            highlight_clip = highlight_clip.crossfadein(0.05).crossfadeout(0.05)
            clips.append(highlight_clip)
            # print(f"Highlighted '{word.word}' at pos_x: {current_x}, start: {word.start}, duration: {word_duration}")
        elif style["highlight-mode"] == "box":
            duration = word.end - word.start
            box_clip = create_box_clip(word, frame_size, (current_x, base_y), duration, style)
            box_clip = box_clip.set_start(word.start)
            clips.append(box_clip)

        current_x += word_width + WORD_SPACING

    return clips

def create_final_video(video_path, transcript, output_path, font_file, style_key="style1"):
    """Create final video with TikTok-style subtitles"""
    try:
        video = VideoFileClip(video_path, audio=True, target_resolution=(720, None))
        frame_size = video.size

        words = adjust_word_timestamps(transcript)
        if not words:
            print("No valid words found in transcript. Skipping...")
            video.close()
            return None

        sentences = group_words_into_sentences(
            words,
            max_words=MAX_WORDS_PER_SENTENCE,
            frame_width=frame_size[0],
            font_size=int(frame_size[1] * SUBTITLE_STYLES[style_key]["font-size"]),
            font_file=font_file,
        )
        if not sentences:
            print("No sentences generated, skipping video creation")
            video.close()
            return None

        all_clips = []
        for sentence in sentences:
            sentence_clips = create_sentence_clips(sentence, frame_size, font_file, style_key)
            all_clips.extend(sentence_clips)

        if not all_clips:
            print("No clips generated, skipping video write")
            video.close()
            return None

        temp_output = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
        final_video = CompositeVideoClip([video] + all_clips, size=frame_size)

        final_video.write_videofile(
            temp_output,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="faster",
            threads=4,
            bitrate="2000k",
        )

        video.close()
        final_video.close()

        import shutil
        shutil.move(temp_output, output_path)
        return output_path

    except Exception as e:
        print(f"Error creating final video: {e}")
        if "video" in locals():
            video.close()
        return None

def process_video(video_file, video_num, font_file, style_key, api_key):
    """Process a single video: extract audio, generate transcript, add subtitles"""
    video_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, f"final_video_{video_num}.mp4")
    temp_audio_path = os.path.join(temp_folder, f"audio_{video_num}.mp3")

    print(f"Processing {video_file} with style {style_key}...")

    audio_path = extract_audio(video_path, temp_audio_path)
    if not audio_path:
        print(f"Failed to extract audio for {video_file}. Skipping...")
        return None

    print(f"Generating transcript for {video_file}...")
    transcript = generate_subtitles(audio_path, api_key)
    if not transcript:
        print(f"Failed to generate transcript for {video_file}. Skipping...")
        os.remove(temp_audio_path)
        return None

    print(f"Creating final video with subtitles for {video_file}...")
    start_time = time.time()
    final_video = create_final_video(video_path, transcript, output_path, font_file, style_key)
    elapsed = time.time() - start_time

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    if final_video:
        print(f"Successfully created video: {output_path} in {elapsed:.2f} seconds")
        return output_path
    else:
        print(f"Failed to create final video for {video_file}")
        return None

def process_subtitles(reframe: bool, style_key="style1"):
    """Process subtitles for all videos in input folder with a single style parameter"""
    global input_folder  # Declare input_folder as global to modify it within the function
    
    start_time = time.time()

    # Set input folder based on reframe parameter
    if reframe:
        input_folder = os.path.join(script_dir, "../s2_curation/assets/processed_shorts/")
    else:
        input_folder = os.path.join(script_dir, "../s2_curation/assets/curated_videos/")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env file. Exiting...")
        return

    style = SUBTITLE_STYLES.get(style_key, SUBTITLE_STYLES["style1"])
    font_file = os.path.join(script_dir, "fonts", style["fontStyle"])
    if not os.path.exists(font_file):
        print(f"Font file not found: {font_file}. Using default 'Chewy-Regular.ttf'")
        font_file = os.path.join(script_dir, "fonts", "Chewy-Regular.ttf")
        if not os.path.exists(font_file):
            print(f"Default font 'Chewy-Regular.ttf' not found either. Please check font files!")
            return

    print(f"Using font: {style['fontStyle']} ({font_file})")
    print(f"Using subtitle style: {style_key}")
    print(f"Input folder set to: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    video_files = [
        f for f in os.listdir(input_folder)
        if f.startswith("curated_vid_") and f.endswith(".mp4")
    ]
    video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    processing_params = []
    for i, video_file in enumerate(video_files):
        video_num = video_file.split("_")[-1].split(".")[0]
        processing_params.append((video_file, video_num, font_file, style_key, api_key))

    max_workers = min(os.cpu_count(), len(processing_params))
    print(f"Processing {len(processing_params)} videos with {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_video, *params) for params in processing_params
        ]
        completed_videos = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    elapsed = time.time() - start_time
    print(f"All videos processed in {elapsed:.2f} seconds")

import numpy as np

if __name__ == "__main__":
    process_subtitles(reframe=False, style_key="style1")