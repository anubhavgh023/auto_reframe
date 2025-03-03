from s1_yt_down.yt_download import download_video
from s2_curation.curation_processing import curation_processing 
from s3_obj_detect_crop.detect_reframe import (generate_transitions_json,process_videos)
from s4_add_subtitle.sub import process_subtitles 
from s5_bg_music_adder.bg_adder  import process_background_music 
from helpers.cleaner import clean_files

def main():
    # #step 1:
    # video_url = "https://www.youtube.com/watch?v=XXYNvwrVKdE"
    # download_video(video_url)

    #step 2:
    print("Processing curation...")
    curation_processing()

    #step 3:
    print("Generating transition data...")
    generate_transitions_json()

    print("\nProcessing videos...")
    process_videos()

    print("\nAll videos processed successfully!")

    #step 4:
    process_subtitles()

    #step 5:
    process_background_music()

    # Step 6: Clean up temporary files
    # clean_files()

if __name__ == "__main__":
    main()
