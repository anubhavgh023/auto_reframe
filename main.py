# from s1_yt_down.yt_download import download_video
# from s2_curation.curation_processing import curation_processing
# from s3_obj_detect_crop.detect_reframe import (generate_transitions_json,process_videos)
# from s4_add_subtitle.sub import process_subtitles
# from s5_bg_music_adder.bg_adder  import process_background_music
# from helpers.cleaner import clean_files

# def main():
#     # #step 1:
#     video_url = "https://www.youtube.com/watch?v=3A8kawxMOcQ"
#     download_video(video_url)

#     # step 2:
#     print("Processing curation...")
#     curation_processing()

#     # step 3:
#     print("Generating transition data...")
#     generate_transitions_json()

#     print("\nProcessing videos...")
#     process_videos()

#     print("\nAll videos processed successfully!")

#     # step 4:
#     process_subtitles()

#     # step 5:
#     process_background_music()

#     # Step 6: Clean up temporary files
#     clean_files()

# if __name__ == "__main__":
#     main()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# Import your existing functions
from s1_yt_down.yt_download import download_video
from s2_curation.curation_processing import curation_processing 
from s3_obj_detect_crop.detect_reframe import (generate_transitions_json, process_videos)
from s4_add_subtitle.sub import process_subtitles 
from s5_bg_music_adder.bg_adder import process_background_music 
from helpers.cleaner import clean_files

app = FastAPI()

# Define request body model
class VideoRequest(BaseModel):
    youtubeLink: str
    duration: int
    numOfshorts: int 
    language: str
    fontStyle: str

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        # Validate duration
        if request.duration not in [45, 60]:
            raise HTTPException(
                status_code=400,
                detail="Duration must be either 45 or 60 seconds"
            )

        if request.numOfshorts > 5:
            raise HTTPException(
                status_code=400,
                detail="Num of shorts greater than 5"
            )

        # Print request:
        print("----------------------------------")
        print(f"req: {request}")
        print("----------------------------------")

        # Step 1: Download video
        try:
            download_video(request.youtubeLink)
        except Exception as download_error:
            if "Video exceeds 45-minute time limit" in str(download_error):
                raise HTTPException(
                    status_code=400,
                    detail="Video duration exceeds 45-minute limit"
                )
            raise  # Re-raise other download errors

        # # Step 2: Process curation
        print("Processing curation...")
        curation_processing(duration=request.duration,numOfShorts=request.numOfshorts)

        # # Step 3: Generate transitions and process videos
        print("Generating transition data...")
        generate_transitions_json()
        print("\nProcessing videos...")
        process_videos()

        # # Step 4: Process subtitles
        # You might want to pass fontStyle and language to this function
        print(f"Processing subtitles with font: {request.fontStyle}")
        process_subtitles(request.fontStyle)

        # # Step 5: Add background music
        process_background_music()

        # # Step 6: Clean up
        clean_files()

        return {
            "status": "success",
            "message": "Video processed successfully",
            "youtube_link": request.youtubeLink,
            "duration": request.duration,
            "language": request.language,
            "font_style": request.fontStyle,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Video Processing API"}
