from s2_curation.s1_audio_extracting import extract_audio 
from s2_curation.s2_chunk_aud import chunk_audio
from s2_curation.s3_transcript_chunker import (
    process_chunks,adjust_transcript_timestamps
)
from s2_curation.s4_sliding_window_transcript import generate_sliding_transcripts
from s2_curation.s5_score_extractor import process_transcript_main 
from s2_curation.s6_find_curated_video import curation
from s2_curation.s7_vid_extractor import extract_video_segments
import os

# testing
# from s1_audio_extracting import extract_audio 
# from s2_chunk_aud import chunk_audio
# from s3_transcript_chunker import (
#     process_chunks,adjust_transcript_timestamps
# )
# from s4_sliding_window_transcript import generate_sliding_transcripts
# from s5_score_extractor import process_transcript_main 
# from s6_find_curated_video import curation
# from s7_vid_extractor import extract_video_segments
# import os


def curation_processing(duration:int,numOfShorts:int):
    # Step 1: Extract audio from input video
    # extract_audio("../downloads/input_video.mp4", "./assets/audio/full_audio.wav")

   # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path to the input video
    video_path = os.path.join(script_dir, "../downloads/input_video.mp4")
    video_path = os.path.abspath(video_path)  # Resolve to absolute path
    
    # Construct the absolute path to the output audio file
    audio_output_path = os.path.join(script_dir, "./assets/audio/full_audio.wav")
    audio_output_path = os.path.abspath(audio_output_path)  # Resolve to absolute path
    
    # Step 1: Extract audio from input video
    extract_audio(video_path, audio_output_path)
    
    # Step 2: chunk the audio file 
    chunk_audio(duration)
    
    # Step 3: Extract frames and process chunks
    if process_chunks():
        # Only adjust timestamps if processing was successful
        adjust_transcript_timestamps()
        print("\nAll processing completed!")
    else:
        print("\nNo successful transcriptions to adjust timestamps for.")
    
    # Step 4: Sliding window processing
    generate_sliding_transcripts()
    
    # Step 5: Process the transcript files
    process_transcript_main()
    
    # Step 6: Extract Video Transcript Segments based on score 
    curation(numOfShorts=numOfShorts)

    # Step 7: Extract/crop out the selected video segments using ffmpeg
    extract_video_segments()

if __name__ == "__main__":
    curation_processing(60,2)
