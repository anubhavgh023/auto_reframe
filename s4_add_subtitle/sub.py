# import os
# import json
# from dotenv import load_dotenv
# from moviepy.editor import (
#     TextClip,
#     CompositeVideoClip,
#     VideoFileClip,
#     ColorClip,
#     AudioFileClip,
# )

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Input and output paths
# input_folder = os.path.join(script_dir, "../s2_curation/assets/processed_shorts/")
# output_folder = os.path.join(script_dir, "../downloads/")
# transcripts_folder = os.path.join(script_dir, "../s2_curation/assets/audio/sliding_transcript_chunks/")
# mapping_file = os.path.join(script_dir, "../s2_curation/assets/temp/curated_video_segments.json")

# # Global Configuration
# FONT = os.path.join(script_dir, "fonts/Poppins-ExtraBold.ttf")
# FONT_SIZE_RATIO = 0.045
# HIGHLIGHT_COLOR = "#15ff00"
# BOX_COLOR = (0, 77, 170) # dark blue
# STROKE_COLOR = "black"
# STROKE_WIDTH = 2
# HIGHLIGHT_MODE = "per_word" #ops : per_word | box
# WORD_SPACING = 8
# BASE_TEXT_COLOR = "white"
# MAX_LINE_WIDTH_RATIO = 0.8  # Maximum width of text as ratio of video width
# MAX_CHARS_PER_LINE = 30     # Maximum characters per line for long sentences

# def load_transcript_mapping():
#     """Load the mapping between curated videos and transcript files"""
#     try:
#         with open(mapping_file, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading transcript mapping: {e}")
#         return {}

# def load_transcript(transcript_file):
#     """Load transcript from JSON file"""
#     try:
#         with open(transcript_file, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading transcript file: {e}")
#         return None

# class WordInfo:
#     """Class to store word information similar to OpenAI Whisper output format"""
#     def __init__(self, word, start, end):
#         self.word = word
#         self.start = start
#         self.end = end

# def adjust_word_timestamps(transcript_data, segment_start):
#     """Adjust word timestamps relative to the video segment start time"""
#     words = []
#     for word_data in transcript_data["words"]:
#         # Adjust timestamps relative to the video segment
#         adjusted_start = word_data["start"] - segment_start
#         adjusted_end = word_data["end"] - segment_start

#         # Only include words that have positive timestamps (appear in the video segment)
#         if adjusted_end > 0:
#             # Ensure start time is not negative
#             adjusted_start = max(0, adjusted_start)
#             words.append(WordInfo(word_data["word"], adjusted_start, adjusted_end))

#     return words

# def group_words_into_sentences(words, max_words=3, frame_width=None, font_size=None):
#     """
#     Group words into sentences based on timing, max words, and line width
#     Ensures no line exceeds MAX_LINE_WIDTH_RATIO of frame width
#     """
#     sentences = []
#     current_sentence = []
#     word_count = 0
#     last_end_time = 0
#     current_line_length = 0

#     # For line length estimation
#     check_line_length = frame_width is not None and font_size is not None

#     for word in words:
#         # Check if we should start a new sentence based on:
#         # 1. Max words reached
#         # 2. Significant pause in speech
#         # 3. Current line would be too long (if we have frame dimensions)
#         if (word_count >= max_words or
#             (word.start - last_end_time > 0.7 and word_count > 0) or
#             (check_line_length and current_line_length + len(word.word) > MAX_CHARS_PER_LINE)):

#             if current_sentence:
#                 sentences.append(current_sentence)
#                 current_sentence = []
#                 word_count = 0
#                 current_line_length = 0

#         current_sentence.append(word)
#         word_count += 1
#         last_end_time = word.end
#         current_line_length += len(word.word) + 1  # +1 for space

#     if current_sentence:
#         sentences.append(current_sentence)

#     return sentences

# def create_box_clip(word, frame_size, position, duration):
#     """Create a box background for a word"""
#     padding = 9
#     temp_clip = TextClip(
#         word.word,
#         font=FONT,
#         fontsize=int(frame_size[1] * FONT_SIZE_RATIO),
#         color=BASE_TEXT_COLOR,
#         method="label",
#     )

#     box = ColorClip(
#         size=(int(temp_clip.w + 1.2 * padding), int(temp_clip.h + padding)),
#         color=BOX_COLOR,
#         ismask=False,
#     )

#     box = box.set_position((position[0] - padding / 2, position[1] - padding / 4))
#     box = box.set_duration(duration)
#     box = box.set_opacity(1)

#     temp_clip.close()
#     return box

# def create_sentence_clips(sentence_words, frame_size):
#     """Create clips for a sentence with configurable highlighting"""
#     clips = []
#     font_size = int(frame_size[1] * FONT_SIZE_RATIO)
#     base_y = frame_size[1] * 0.7
#     total_width = 0
#     word_clips = []

#     # Create all word clips to measure total width
#     for word in sentence_words:
#         clip = TextClip(
#             word.word,
#             font=FONT,
#             fontsize=font_size,
#             color=BASE_TEXT_COLOR,
#             stroke_color=STROKE_COLOR,
#             stroke_width=STROKE_WIDTH,
#             method="label",
#         )
#         word_clips.append((word, clip))
#         total_width += clip.w + WORD_SPACING

#     # Check if the sentence is too wide for the frame
#     max_width = frame_size[0] * MAX_LINE_WIDTH_RATIO
#     if total_width > max_width:
#         # Close clips to prevent memory leaks
#         for _, clip in word_clips:
#             clip.close()

#         # Split into multiple lines if too wide
#         mid_point = len(sentence_words) // 2
#         line1_words = sentence_words[:mid_point]
#         line2_words = sentence_words[mid_point:]

#         # Create clips for each line
#         line1_clips = create_sentence_clips(line1_words, frame_size)

#         # Adjust position for second line
#         line2_clips = []
#         line2_word_clips = []
#         line2_total_width = 0

#         for word in line2_words:
#             clip = TextClip(
#                 word.word,
#                 font=FONT,
#                 fontsize=font_size,
#                 color=BASE_TEXT_COLOR,
#                 stroke_color=STROKE_COLOR,
#                 stroke_width=STROKE_WIDTH,
#                 method="label",
#             )
#             line2_word_clips.append((word, clip))
#             line2_total_width += clip.w + WORD_SPACING

#         line2_start_x = (frame_size[0] - line2_total_width) / 2
#         line2_current_x = line2_start_x
#         line2_base_y = base_y + font_size + 10  # Add space between lines

#         for word, clip in line2_word_clips:
#             if HIGHLIGHT_MODE == "box":
#                 try:
#                     duration = float(word.end - word.start)
#                     box_clip = create_box_clip(
#                         word, frame_size, (line2_current_x, line2_base_y), duration
#                     )
#                     box_clip = box_clip.set_start(float(word.start))
#                     line2_clips.append(box_clip)
#                 except Exception as e:
#                     print(f"Error creating box clip for word {word.word}: {e}")

#             white_clip = clip.set_position((line2_current_x, line2_base_y))
#             white_clip = white_clip.set_start(sentence_words[0].start)
#             white_clip = white_clip.set_duration(
#                 sentence_words[-1].end - sentence_words[0].start
#             )
#             line2_clips.append(white_clip)

#             if HIGHLIGHT_MODE == "per_word":
#                 highlight_clip = TextClip(
#                     word.word,
#                     font=FONT,
#                     fontsize=font_size,
#                     color=HIGHLIGHT_COLOR,
#                     stroke_color=STROKE_COLOR,
#                     stroke_width=STROKE_WIDTH,
#                     method="label",
#                 ).set_position((line2_current_x, line2_base_y))

#                 highlight_clip = highlight_clip.set_start(word.start)
#                 highlight_clip = highlight_clip.set_duration(word.end - word.start)
#                 line2_clips.append(highlight_clip)

#             line2_current_x += clip.w + WORD_SPACING

#         line2_clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in line2_clips]
#         return line1_clips + line2_clips

#     # If sentence fits on one line, proceed normally
#     start_x = (frame_size[0] - total_width) / 2
#     current_x = start_x

#     for word, clip in word_clips:
#         if HIGHLIGHT_MODE == "box":
#             try:
#                 duration = float(word.end - word.start)
#                 box_clip = create_box_clip(
#                     word, frame_size, (current_x, base_y), duration
#                 )
#                 box_clip = box_clip.set_start(float(word.start))
#                 clips.append(box_clip)
#             except Exception as e:
#                 print(f"Error creating box clip for word {word.word}: {e}")

#         white_clip = clip.set_position((current_x, base_y))
#         white_clip = white_clip.set_start(sentence_words[0].start)
#         white_clip = white_clip.set_duration(
#             sentence_words[-1].end - sentence_words[0].start
#         )
#         clips.append(white_clip)

#         if HIGHLIGHT_MODE == "per_word":
#             highlight_clip = TextClip(
#                 word.word,
#                 font=FONT,
#                 fontsize=font_size,
#                 color=HIGHLIGHT_COLOR,
#                 stroke_color=STROKE_COLOR,
#                 stroke_width=STROKE_WIDTH,
#                 method="label",
#             ).set_position((current_x, base_y))

#             highlight_clip = highlight_clip.set_start(word.start)
#             highlight_clip = highlight_clip.set_duration(word.end - word.start)
#             clips.append(highlight_clip)

#         current_x += clip.w + WORD_SPACING

#     clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in clips]
#     return clips

# def create_final_video(video_path, words, output_path):
#     """Create final video with TikTok-style subtitles"""
#     try:
#         video = VideoFileClip(video_path)
#         frame_size = video.size
#         font_size = int(frame_size[1] * FONT_SIZE_RATIO)

#         # Group words into sentences with consideration for frame size
#         sentences = group_words_into_sentences(words, max_words=3,
#                                               frame_width=frame_size[0],
#                                               font_size=font_size)
#         all_clips = []

#         for sentence in sentences:
#             sentence_clips = create_sentence_clips(sentence, frame_size)
#             all_clips.extend(sentence_clips)

#         final_video = CompositeVideoClip([video] + all_clips, size=frame_size)
#         final_video.write_videofile(
#             output_path, fps=24, codec="libx264", audio_codec="aac"
#         )

#         video.close()
#         final_video.close()

#         return output_path

#     except Exception as e:
#         print(f"Error creating final video: {e}")
#         return None

# def process_subtitles():
#     # Load environment variables
#     load_dotenv()

#     # Create output directory if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Load mapping between curated videos and transcript files
#     transcript_mapping = load_transcript_mapping()
#     if not transcript_mapping:
#         print("Failed to load transcript mapping. Exiting...")
#         return

#     # Get the list of transcript files in order
#     transcript_files = list(transcript_mapping.keys())

#     # Get all video files in the input folder
#     video_files = [
#         f
#         for f in os.listdir(input_folder)
#         if f.startswith("curated_vid_") and f.endswith(".mp4")
#     ]

#     # Sort video files to ensure correct order
#     video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

#     # Process each video with its corresponding transcript
#     for i, video_file in enumerate(video_files):
#         # If we have more videos than transcripts, stop
#         if i >= len(transcript_files):
#             print(f"No more transcript files available for {video_file}. Skipping...")
#             continue

#         # Get the corresponding transcript file (sequential mapping)
#         transcript_file_name = transcript_files[i]
#         segment_info = transcript_mapping[transcript_file_name]
#         segment_start = segment_info["start"]

#         transcript_file = os.path.join(transcripts_folder, transcript_file_name)

#         # Extract video number from filename
#         video_num = video_file.split("_")[-1].split(".")[0]
#         output_path = os.path.join(output_folder, f"final_video_{video_num}.mp4")
#         video_path = os.path.join(input_folder, video_file)

#         print(f"Processing {video_file}...")
#         print(f"Using transcript from {transcript_file_name}")

#         # Load transcript data
#         transcript_data = load_transcript(transcript_file)
#         if not transcript_data:
#             print(f"Failed to load transcript for {video_file}. Skipping...")
#             continue

#         # Adjust word timestamps relative to the video segment start time
#         words = adjust_word_timestamps(transcript_data, segment_start)

#         # Create final video with subtitles
#         print(f"Creating final video with subtitles for {video_file}...")
#         final_video = create_final_video(video_path, words, output_path)

#         if final_video:
#             print(f"Successfully created video: {output_path}")
#         else:
#             print(f"Failed to create final video for {video_file}")

#     print("All videos processed")

# if __name__ == "__main__":
#     process_subtitles()


# ----------------------------------------------- Parallel Code -----------------------------------------------
# import os
# import json
# from dotenv import load_dotenv
# import concurrent.futures
# from moviepy.editor import (
#     TextClip,
#     CompositeVideoClip,
#     VideoFileClip,
#     ColorClip,
# )
# import moviepy.video.fx.all  # Add this import for the multiprocessing setting
# import tempfile
# import time
# from functools import lru_cache

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Input and output paths
# input_folder = os.path.join(script_dir, "../s2_curation/assets/processed_shorts/")
# output_folder = os.path.join(script_dir, "../downloads/")
# transcripts_folder = os.path.join(
#     script_dir, "../s2_curation/assets/audio/sliding_transcript_chunks/"
# )
# mapping_file = os.path.join(
#     script_dir, "../s2_curation/assets/temp/curated_video_segments.json"
# )

# # Global Configuration
# FONT = os.path.join(script_dir, "fonts/Poppins-ExtraBold.ttf")
# FONT_SIZE_RATIO = 0.045
# HIGHLIGHT_COLOR = "#15ff00"
# BOX_COLOR = (0, 77, 170)  # dark blue
# STROKE_COLOR = "black"
# STROKE_WIDTH = 2
# HIGHLIGHT_MODE = "per_word"  # ops : per_word | box
# WORD_SPACING = 8
# BASE_TEXT_COLOR = "white"
# MAX_LINE_WIDTH_RATIO = 0.8  # Maximum width of text as ratio of video width
# MAX_CHARS_PER_LINE = 30  # Maximum characters per line for long sentences

# # Set MoviePy's global parameters for better performance
# import moviepy.config as mpconfig

# mpconfig.FFMPEG_BINARY = "ffmpeg"  # Make sure you have ffmpeg installed
# moviepy.video.fx.all.speedx.use_multiprocessing = (
#     True  # Enable multiprocessing for effects
# )


# # Cache for text clips to avoid recreating identical clips
# @lru_cache(maxsize=512)
# def get_text_dimensions(word, font_size):
#     """Get text dimensions without creating full clip (cached)"""
#     temp_clip = TextClip(
#         word,
#         font=FONT,
#         fontsize=font_size,
#         color=BASE_TEXT_COLOR,
#         method="label",
#     )
#     dimensions = (temp_clip.w, temp_clip.h)
#     temp_clip.close()
#     return dimensions


# def load_transcript_mapping():
#     """Load the mapping between curated videos and transcript files"""
#     try:
#         with open(mapping_file, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading transcript mapping: {e}")
#         return {}


# def load_transcript(transcript_file):
#     """Load transcript from JSON file"""
#     try:
#         with open(transcript_file, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading transcript file: {e}")
#         return None


# class WordInfo:
#     """Class to store word information similar to OpenAI Whisper output format"""

#     def __init__(self, word, start, end):
#         self.word = word
#         self.start = start
#         self.end = end


# def adjust_word_timestamps(transcript_data, segment_start):
#     """Adjust word timestamps relative to the video segment start time"""
#     words = []
#     for word_data in transcript_data["words"]:
#         # Adjust timestamps relative to the video segment
#         adjusted_start = word_data["start"] - segment_start
#         adjusted_end = word_data["end"] - segment_start

#         # Only include words that have positive timestamps (appear in the video segment)
#         if adjusted_end > 0:
#             # Ensure start time is not negative
#             adjusted_start = max(0, adjusted_start)
#             words.append(WordInfo(word_data["word"], adjusted_start, adjusted_end))

#     return words


# def group_words_into_sentences(words, max_words=3, frame_width=None, font_size=None):
#     """
#     Group words into sentences based on timing, max words, and line width
#     Ensures no line exceeds MAX_LINE_WIDTH_RATIO of frame width
#     """
#     sentences = []
#     current_sentence = []
#     word_count = 0
#     last_end_time = 0
#     current_line_length = 0

#     # For line length estimation
#     check_line_length = frame_width is not None and font_size is not None

#     for word in words:
#         # Check if we should start a new sentence based on:
#         # 1. Max words reached
#         # 2. Significant pause in speech
#         # 3. Current line would be too long (if we have frame dimensions)
#         if (
#             word_count >= max_words
#             or (word.start - last_end_time > 0.7 and word_count > 0)
#             or (
#                 check_line_length
#                 and current_line_length + len(word.word) > MAX_CHARS_PER_LINE
#             )
#         ):

#             if current_sentence:
#                 sentences.append(current_sentence)
#                 current_sentence = []
#                 word_count = 0
#                 current_line_length = 0

#         current_sentence.append(word)
#         word_count += 1
#         last_end_time = word.end
#         current_line_length += len(word.word) + 1  # +1 for space

#     if current_sentence:
#         sentences.append(current_sentence)

#     return sentences


# def create_box_clip(word, frame_size, position, duration):
#     """Create a box background for a word"""
#     padding = 9
#     word_w, word_h = get_text_dimensions(
#         word.word, int(frame_size[1] * FONT_SIZE_RATIO)
#     )

#     box = ColorClip(
#         size=(int(word_w + 1.2 * padding), int(word_h + padding)),
#         color=BOX_COLOR,
#         ismask=False,
#     )

#     box = box.set_position((position[0] - padding / 2, position[1] - padding / 4))
#     box = box.set_duration(duration)
#     box = box.set_opacity(1)

#     return box


# def create_sentence_clips(sentence_words, frame_size):
#     """Create clips for a sentence with configurable highlighting (optimized)"""
#     clips = []
#     font_size = int(frame_size[1] * FONT_SIZE_RATIO)
#     base_y = frame_size[1] * 0.7
#     total_width = 0
#     word_dimensions = []

#     # Calculate total width without creating clips
#     for word in sentence_words:
#         w, h = get_text_dimensions(word.word, font_size)
#         word_dimensions.append((word, w, h))
#         total_width += w + WORD_SPACING

#     # Check if the sentence is too wide for the frame
#     max_width = frame_size[0] * MAX_LINE_WIDTH_RATIO
#     if total_width > max_width:
#         # Split into multiple lines if too wide
#         mid_point = len(sentence_words) // 2
#         line1_words = sentence_words[:mid_point]
#         line2_words = sentence_words[mid_point:]

#         # Create clips for each line
#         line1_clips = create_sentence_clips(line1_words, frame_size)

#         # Adjust position for second line
#         line2_clips = []
#         line2_word_dimensions = []
#         line2_total_width = 0

#         for word in line2_words:
#             w, h = get_text_dimensions(word.word, font_size)
#             line2_word_dimensions.append((word, w, h))
#             line2_total_width += w + WORD_SPACING

#         line2_start_x = (frame_size[0] - line2_total_width) / 2
#         line2_current_x = line2_start_x
#         line2_base_y = base_y + font_size + 10  # Add space between lines

#         for word, w, h in line2_word_dimensions:
#             if HIGHLIGHT_MODE == "box":
#                 try:
#                     duration = float(word.end - word.start)
#                     box_clip = create_box_clip(
#                         word, frame_size, (line2_current_x, line2_base_y), duration
#                     )
#                     box_clip = box_clip.set_start(float(word.start))
#                     line2_clips.append(box_clip)
#                 except Exception as e:
#                     print(f"Error creating box clip for word {word.word}: {e}")

#             # Create white clip for all words in the sentence
#             white_clip = TextClip(
#                 word.word,
#                 font=FONT,
#                 fontsize=font_size,
#                 color=BASE_TEXT_COLOR,
#                 stroke_color=STROKE_COLOR,
#                 stroke_width=STROKE_WIDTH,
#                 method="label",
#             ).set_position((line2_current_x, line2_base_y))

#             white_clip = white_clip.set_start(sentence_words[0].start)
#             white_clip = white_clip.set_duration(
#                 sentence_words[-1].end - sentence_words[0].start
#             )
#             line2_clips.append(white_clip)

#             if HIGHLIGHT_MODE == "per_word":
#                 highlight_clip = TextClip(
#                     word.word,
#                     font=FONT,
#                     fontsize=font_size,
#                     color=HIGHLIGHT_COLOR,
#                     stroke_color=STROKE_COLOR,
#                     stroke_width=STROKE_WIDTH,
#                     method="label",
#                 ).set_position((line2_current_x, line2_base_y))

#                 highlight_clip = highlight_clip.set_start(word.start)
#                 highlight_clip = highlight_clip.set_duration(word.end - word.start)
#                 line2_clips.append(highlight_clip)

#             line2_current_x += w + WORD_SPACING

#         # Only apply crossfade once after creating all clips to reduce overhead
#         line2_clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in line2_clips]
#         return line1_clips + line2_clips

#     # If sentence fits on one line, proceed normally
#     start_x = (frame_size[0] - total_width) / 2
#     current_x = start_x

#     for word, w, h in word_dimensions:
#         if HIGHLIGHT_MODE == "box":
#             try:
#                 duration = float(word.end - word.start)
#                 box_clip = create_box_clip(
#                     word, frame_size, (current_x, base_y), duration
#                 )
#                 box_clip = box_clip.set_start(float(word.start))
#                 clips.append(box_clip)
#             except Exception as e:
#                 print(f"Error creating box clip for word {word.word}: {e}")

#         white_clip = TextClip(
#             word.word,
#             font=FONT,
#             fontsize=font_size,
#             color=BASE_TEXT_COLOR,
#             stroke_color=STROKE_COLOR,
#             stroke_width=STROKE_WIDTH,
#             method="label",
#         ).set_position((current_x, base_y))

#         white_clip = white_clip.set_start(sentence_words[0].start)
#         white_clip = white_clip.set_duration(
#             sentence_words[-1].end - sentence_words[0].start
#         )
#         clips.append(white_clip)

#         if HIGHLIGHT_MODE == "per_word":
#             highlight_clip = TextClip(
#                 word.word,
#                 font=FONT,
#                 fontsize=font_size,
#                 color=HIGHLIGHT_COLOR,
#                 stroke_color=STROKE_COLOR,
#                 stroke_width=STROKE_WIDTH,
#                 method="label",
#             ).set_position((current_x, base_y))

#             highlight_clip = highlight_clip.set_start(word.start)
#             highlight_clip = highlight_clip.set_duration(word.end - word.start)
#             clips.append(highlight_clip)

#         current_x += w + WORD_SPACING

#     # Apply crossfades to all clips at once instead of one by one
#     clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in clips]
#     return clips


# def create_final_video(video_path, words, output_path):
#     """Create final video with TikTok-style subtitles (optimized)"""
#     try:
#         # Load video more efficiently with lower quality during processing
#         video = VideoFileClip(video_path, audio=True, target_resolution=(720, None))
#         frame_size = video.size
#         font_size = int(frame_size[1] * FONT_SIZE_RATIO)

#         # Group words into sentences with consideration for frame size
#         sentences = group_words_into_sentences(
#             words, max_words=3, frame_width=frame_size[0], font_size=font_size
#         )
#         all_clips = []

#         # Create clips for all sentences
#         for sentence in sentences:
#             sentence_clips = create_sentence_clips(sentence, frame_size)
#             all_clips.extend(sentence_clips)

#         # Create final video with lower bitrate during processing
#         temp_output = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
#         final_video = CompositeVideoClip([video] + all_clips, size=frame_size)

#         # Write with optimized parameters
#         final_video.write_videofile(
#             temp_output,
#             fps=24,
#             codec="libx264",
#             audio_codec="aac",
#             preset="faster",  # Faster encoding
#             threads=4,  # Use multiple threads
#             bitrate="2000k",  # Lower bitrate for faster processing
#         )

#         # Close clips to prevent memory leaks
#         video.close()
#         final_video.close()

#         # Move from temp location to final destination
#         import shutil

#         shutil.move(temp_output, output_path)

#         return output_path

#     except Exception as e:
#         print(f"Error creating final video: {e}")
#         return None


# def process_video(video_file, transcript_file_name, segment_start, video_num):
#     """Process a single video with its transcript (for parallel processing)"""
#     output_path = os.path.join(output_folder, f"final_video_{video_num}.mp4")
#     video_path = os.path.join(input_folder, video_file)
#     transcript_file = os.path.join(transcripts_folder, transcript_file_name)

#     print(f"Processing {video_file}...")
#     print(f"Using transcript from {transcript_file_name}")

#     # Load transcript data
#     transcript_data = load_transcript(transcript_file)
#     if not transcript_data:
#         print(f"Failed to load transcript for {video_file}. Skipping...")
#         return None

#     # Adjust word timestamps relative to the video segment start time
#     words = adjust_word_timestamps(transcript_data, segment_start)

#     # Create final video with subtitles
#     print(f"Creating final video with subtitles for {video_file}...")
#     start_time = time.time()
#     final_video = create_final_video(video_path, words, output_path)
#     elapsed = time.time() - start_time

#     if final_video:
#         print(f"Successfully created video: {output_path} in {elapsed:.2f} seconds")
#         return output_path
#     else:
#         print(f"Failed to create final video for {video_file}")
#         return None


# def process_subtitles():
#     start_time = time.time()

#     # Load environment variables
#     load_dotenv()

#     # Create output directory if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Load mapping between curated videos and transcript files
#     transcript_mapping = load_transcript_mapping()
#     if not transcript_mapping:
#         print("Failed to load transcript mapping. Exiting...")
#         return

#     # Get the list of transcript files in order
#     transcript_files = list(transcript_mapping.keys())

#     # Get all video files in the input folder
#     video_files = [
#         f
#         for f in os.listdir(input_folder)
#         if f.startswith("curated_vid_") and f.endswith(".mp4")
#     ]

#     # Sort video files to ensure correct order
#     video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

#     # Prepare parameters for parallel processing
#     processing_params = []
#     for i, video_file in enumerate(video_files):
#         # If we have more videos than transcripts, stop
#         if i >= len(transcript_files):
#             print(f"No more transcript files available for {video_file}. Skipping...")
#             continue

#         # Get the corresponding transcript file (sequential mapping)
#         transcript_file_name = transcript_files[i]
#         segment_info = transcript_mapping[transcript_file_name]
#         segment_start = segment_info["start"]

#         # Extract video number from filename
#         video_num = video_file.split("_")[-1].split(".")[0]

#         # Add to processing parameters
#         processing_params.append(
#             (video_file, transcript_file_name, segment_start, video_num)
#         )

#     # Process videos in parallel using a thread pool
#     max_workers = min(os.cpu_count(), len(processing_params))
#     print(f"Processing {len(processing_params)} videos with {max_workers} workers")

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(process_video, *params) for params in processing_params
#         ]

#         # Wait for all futures to complete
#         completed_videos = [
#             future.result() for future in concurrent.futures.as_completed(futures)
#         ]

#     elapsed = time.time() - start_time
#     print(f"All videos processed in {elapsed:.2f} seconds")

# if __name__ == "__main__":
#     process_subtitles()

# ------------------------------ Parallel with FONT selection -------------------------------
import os
import json
from dotenv import load_dotenv
import concurrent.futures
from moviepy.editor import (
    TextClip,
    CompositeVideoClip,
    VideoFileClip,
    ColorClip,
)
import moviepy.video.fx.all
import tempfile
import time
from functools import lru_cache

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input and output paths
input_folder = os.path.join(script_dir, "../s2_curation/assets/processed_shorts/")
output_folder = os.path.join(script_dir, "../downloads/")
transcripts_folder = os.path.join(
    script_dir, "../s2_curation/assets/audio/sliding_transcript_chunks/"
)
mapping_file = os.path.join(
    script_dir, "../s2_curation/assets/temp/curated_video_segments.json"
)

# Global Font Dictionary
FONTS = {
    "montserrat": "Montserrat-ExtraBold.ttf",
    "bebas-neue": "BebasNeue-Regular.ttf",
    "roboto": "Roboto-Bold.ttf",
    "oswald": "Oswald-SemiBold.ttf",
    "poppins": "Poppins-ExtraBold.ttf",
    "open-sans-extra-bold": "OpenSans-ExtraBold.ttf",
}

# Global Configuration
FONT_SIZE_RATIO = 0.045
HIGHLIGHT_COLOR = "#15ff00"
BOX_COLOR = (0, 77, 170)  # dark blue
STROKE_COLOR = "black"
STROKE_WIDTH = 2
HIGHLIGHT_MODE = "per_word"  # ops : per_word | box
WORD_SPACING = 8
BASE_TEXT_COLOR = "white"
MAX_LINE_WIDTH_RATIO = 0.8
MAX_CHARS_PER_LINE = 30

# Set MoviePy's global parameters for better performance
import moviepy.config as mpconfig

mpconfig.FFMPEG_BINARY = "ffmpeg"
moviepy.video.fx.all.speedx.use_multiprocessing = True


@lru_cache(maxsize=512)
def get_text_dimensions(word, font_size, font_file):
    """Get text dimensions without creating full clip (cached)"""
    temp_clip = TextClip(
        word,
        font=font_file,
        fontsize=font_size,
        color=BASE_TEXT_COLOR,
        method="label",
    )
    dimensions = (temp_clip.w, temp_clip.h)
    temp_clip.close()
    return dimensions


def load_transcript_mapping():
    """Load the mapping between curated videos and transcript files"""
    try:
        with open(mapping_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading transcript mapping: {e}")
        return {}


def load_transcript(transcript_file):
    """Load transcript from JSON file"""
    try:
        with open(transcript_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading transcript file: {e}")
        return None


class WordInfo:
    """Class to store word information similar to OpenAI Whisper output format"""

    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


def adjust_word_timestamps(transcript_data, segment_start):
    """Adjust word timestamps relative to the video segment start time"""
    words = []
    for word_data in transcript_data["words"]:
        adjusted_start = word_data["start"] - segment_start
        adjusted_end = word_data["end"] - segment_start
        if adjusted_end > 0:
            adjusted_start = max(0, adjusted_start)
            words.append(WordInfo(word_data["word"], adjusted_start, adjusted_end))
    return words


def group_words_into_sentences(
    words, max_words=3, frame_width=None, font_size=None, font_file=None
):
    """Group words into sentences based on timing, max words, and line width"""
    sentences = []
    current_sentence = []
    word_count = 0
    last_end_time = 0
    current_line_length = 0
    check_line_length = (
        frame_width is not None and font_size is not None and font_file is not None
    )

    for word in words:
        if (
            word_count >= max_words
            or (word.start - last_end_time > 0.7 and word_count > 0)
            or (
                check_line_length
                and current_line_length + len(word.word) > MAX_CHARS_PER_LINE
            )
        ):
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
                word_count = 0
                current_line_length = 0

        current_sentence.append(word)
        word_count += 1
        last_end_time = word.end
        current_line_length += len(word.word) + 1

    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def create_box_clip(word, frame_size, position, duration, font_file):
    """Create a box background for a word"""
    padding = 9
    word_w, word_h = get_text_dimensions(
        word.word, int(frame_size[1] * FONT_SIZE_RATIO), font_file
    )
    box = ColorClip(
        size=(int(word_w + 1.2 * padding), int(word_h + padding)),
        color=BOX_COLOR,
        ismask=False,
    )
    box = box.set_position((position[0] - padding / 2, position[1] - padding / 4))
    box = box.set_duration(duration)
    box = box.set_opacity(1)
    return box


def create_sentence_clips(sentence_words, frame_size, font_file):
    """Create clips for a sentence with configurable highlighting"""
    clips = []
    font_size = int(frame_size[1] * FONT_SIZE_RATIO)
    base_y = frame_size[1] * 0.7
    total_width = 0
    word_dimensions = []

    for word in sentence_words:
        w, h = get_text_dimensions(word.word, font_size, font_file)
        word_dimensions.append((word, w, h))
        total_width += w + WORD_SPACING

    max_width = frame_size[0] * MAX_LINE_WIDTH_RATIO
    if total_width > max_width:
        mid_point = len(sentence_words) // 2
        line1_words = sentence_words[:mid_point]
        line2_words = sentence_words[mid_point:]
        line1_clips = create_sentence_clips(line1_words, frame_size, font_file)

        line2_clips = []
        line2_word_dimensions = []
        line2_total_width = 0
        for word in line2_words:
            w, h = get_text_dimensions(word.word, font_size, font_file)
            line2_word_dimensions.append((word, w, h))
            line2_total_width += w + WORD_SPACING

        line2_start_x = (frame_size[0] - line2_total_width) / 2
        line2_current_x = line2_start_x
        line2_base_y = base_y + font_size + 10

        for word, w, h in line2_word_dimensions:
            if HIGHLIGHT_MODE == "box":
                try:
                    duration = float(word.end - word.start)
                    box_clip = create_box_clip(
                        word,
                        frame_size,
                        (line2_current_x, line2_base_y),
                        duration,
                        font_file,
                    )
                    box_clip = box_clip.set_start(float(word.start))
                    line2_clips.append(box_clip)
                except Exception as e:
                    print(f"Error creating box clip for word {word.word}: {e}")

            white_clip = TextClip(
                word.word,
                font=font_file,
                fontsize=font_size,
                color=BASE_TEXT_COLOR,
                stroke_color=STROKE_COLOR,
                stroke_width=STROKE_WIDTH,
                method="label",
            ).set_position((line2_current_x, line2_base_y))
            white_clip = white_clip.set_start(sentence_words[0].start)
            white_clip = white_clip.set_duration(
                sentence_words[-1].end - sentence_words[0].start
            )
            line2_clips.append(white_clip)

            if HIGHLIGHT_MODE == "per_word":
                highlight_clip = TextClip(
                    word.word,
                    font=font_file,
                    fontsize=font_size,
                    color=HIGHLIGHT_COLOR,
                    stroke_color=STROKE_COLOR,
                    stroke_width=STROKE_WIDTH,
                    method="label",
                ).set_position((line2_current_x, line2_base_y))
                highlight_clip = highlight_clip.set_start(word.start)
                highlight_clip = highlight_clip.set_duration(word.end - word.start)
                line2_clips.append(highlight_clip)

            line2_current_x += w + WORD_SPACING

        line2_clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in line2_clips]
        return line1_clips + line2_clips

    start_x = (frame_size[0] - total_width) / 2
    current_x = start_x

    for word, w, h in word_dimensions:
        if HIGHLIGHT_MODE == "box":
            try:
                duration = float(word.end - word.start)
                box_clip = create_box_clip(
                    word, frame_size, (current_x, base_y), duration, font_file
                )
                box_clip = box_clip.set_start(float(word.start))
                clips.append(box_clip)
            except Exception as e:
                print(f"Error creating box clip for word {word.word}: {e}")

        white_clip = TextClip(
            word.word,
            font=font_file,
            fontsize=font_size,
            color=BASE_TEXT_COLOR,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method="label",
        ).set_position((current_x, base_y))
        white_clip = white_clip.set_start(sentence_words[0].start)
        white_clip = white_clip.set_duration(
            sentence_words[-1].end - sentence_words[0].start
        )
        clips.append(white_clip)

        if HIGHLIGHT_MODE == "per_word":
            highlight_clip = TextClip(
                word.word,
                font=font_file,
                fontsize=font_size,
                color=HIGHLIGHT_COLOR,
                stroke_color=STROKE_COLOR,
                stroke_width=STROKE_WIDTH,
                method="label",
            ).set_position((current_x, base_y))
            highlight_clip = highlight_clip.set_start(word.start)
            highlight_clip = highlight_clip.set_duration(word.end - word.start)
            clips.append(highlight_clip)

        current_x += w + WORD_SPACING

    clips = [clip.crossfadein(0.1).crossfadeout(0.1) for clip in clips]
    return clips


def create_final_video(video_path, words, output_path, font_file):
    """Create final video with TikTok-style subtitles"""
    try:
        video = VideoFileClip(video_path, audio=True, target_resolution=(720, None))
        frame_size = video.size
        font_size = int(frame_size[1] * FONT_SIZE_RATIO)

        sentences = group_words_into_sentences(
            words,
            max_words=3,
            frame_width=frame_size[0],
            font_size=font_size,
            font_file=font_file,
        )
        all_clips = []

        for sentence in sentences:
            sentence_clips = create_sentence_clips(sentence, frame_size, font_file)
            all_clips.extend(sentence_clips)

        temp_output = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
        final_video = CompositeVideoClip([video] + all_clips, size=frame_size)

        final_video.write_videofile(
            temp_output,
            fps=24,
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
        return None


def process_video(
    video_file, transcript_file_name, segment_start, video_num, font_file
):
    """Process a single video with its transcript"""
    output_path = os.path.join(output_folder, f"final_video_{video_num}.mp4")
    video_path = os.path.join(input_folder, video_file)
    transcript_file = os.path.join(transcripts_folder, transcript_file_name)

    print(f"Processing {video_file}...")
    print(f"Using transcript from {transcript_file_name}")

    transcript_data = load_transcript(transcript_file)
    if not transcript_data:
        print(f"Failed to load transcript for {video_file}. Skipping...")
        return None

    words = adjust_word_timestamps(transcript_data, segment_start)

    print(f"Creating final video with subtitles for {video_file}...")
    start_time = time.time()
    final_video = create_final_video(video_path, words, output_path, font_file)
    elapsed = time.time() - start_time

    if final_video:
        print(f"Successfully created video: {output_path} in {elapsed:.2f} seconds")
        return output_path
    else:
        print(f"Failed to create final video for {video_file}")
        return None


def process_subtitles(font_style="poppins"):
    """
    Process subtitles with a selected font style.

    Args:
        font_style (str): Font style key from FONTS dictionary (default: "poppins")
    """
    start_time = time.time()

    # Validate and get font file
    font_file = FONTS.get(font_style.lower())
    if not font_file:
        print(f"Invalid font style: {font_style}. Using default 'poppins'")
        font_file = FONTS["poppins"]

    font_file = os.path.join(script_dir, "fonts", font_file)
    if not os.path.exists(font_file):
        print(f"Font file not found: {font_file}. Using default 'poppins'")
        font_file = os.path.join(script_dir, "fonts", FONTS["poppins"])

    print(f"Using font: {font_style} ({font_file})")

    load_dotenv()
    os.makedirs(output_folder, exist_ok=True)

    transcript_mapping = load_transcript_mapping()
    if not transcript_mapping:
        print("Failed to load transcript mapping. Exiting...")
        return

    transcript_files = list(transcript_mapping.keys())
    video_files = [
        f
        for f in os.listdir(input_folder)
        if f.startswith("curated_vid_") and f.endswith(".mp4")
    ]
    video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    processing_params = []
    for i, video_file in enumerate(video_files):
        if i >= len(transcript_files):
            print(f"No more transcript files available for {video_file}. Skipping...")
            continue

        transcript_file_name = transcript_files[i]
        segment_info = transcript_mapping[transcript_file_name]
        segment_start = segment_info["start"]
        video_num = video_file.split("_")[-1].split(".")[0]

        processing_params.append(
            (video_file, transcript_file_name, segment_start, video_num, font_file)
        )

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


if __name__ == "__main__":
    process_subtitles()