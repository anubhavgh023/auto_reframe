# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Literal

# # Import your existing functions
# from s1_yt_down.yt_download import download_video
# from s2_curation.curation_processing import curation_processing 
# from s3_obj_detect_crop.detect_reframe import (generate_transitions_json, process_videos)
# from s4_add_subtitle.sub import process_subtitles 
# from s5_bg_music_adder.bg_adder import process_background_music 
# from helpers.cleaner import clean_files

# app = FastAPI()

# # Define request body model
# class VideoRequest(BaseModel):
#     youtubeLink: str
#     duration: int
#     numOfshorts: int 
#     reframe: bool
#     language: str
#     fontStyle: str
#     captions: bool
#     bg_music: str


# @app.post("/process-video")
# async def process_video(request: VideoRequest):
#     try:
#         # Validate duration
#         if request.duration not in [45, 60]:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Duration must be either 45 or 60 seconds"
#             )

#         if request.numOfshorts > 5:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Num of shorts greater than 5"
#             )

#         # Print request:
#         print("----------------------------------")
#         print(f"req: {request}")
#         print("----------------------------------")

#         # Step 1: Download video
#         try:
#             print("----------------------------------")
#             print("STEP 2: Processing curation...")
#             download_video(request.youtubeLink)
#             print("----------------------------------")
#         except Exception as download_error:
#             if "Video exceeds 45-minute time limit" in str(download_error):
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Video duration exceeds 45-minute limit"
#                 )
#             raise  # Re-raise other download errors

#         # Step 2: Process curation
#         print("----------------------------------")
#         print("STEP 2: Processing curation...")
#         print("----------------------------------")
#         curation_processing(duration=request.duration,numOfShorts=request.numOfshorts)

#         # # Step 3: Generate transitions and process videos
#         print("----------------------------------")
#         print("STEP 3-a: Generating transition data...")
#         print("----------------------------------")
#         generate_transitions_json()

#         print("----------------------------------")
#         print("\n STEP 3-b: 3Processing videos...")
#         print("----------------------------------")
#         process_videos()

#         # # Step 4: Process subtitles
#         # You might want to pass fontStyle and language to this function
#         print("----------------------------------")
#         print(f"STEP 4: Processing subtitles with font: {request.fontStyle}")
#         print("----------------------------------")
#         process_subtitles(request.fontStyle)

#         # Step 5: Add background music
#         print("----------------------------------")
#         print("STEP 5: Adding bg music")
#         print("----------------------------------")
#         process_background_music()

#         # # Step 6: Clean up
#         # clean_files()

#         return {
#             "status": "success",
#             "youtube_link": request.youtubeLink,
#             "duration": request.duration,
#             "language": request.language,
#             "font_style": request.fontStyle,
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Optional: Add a root endpoint
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Video Processing API"}


# --------------------------------


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
import os
import glob

# Import your existing functions
from s1_yt_down.yt_download import download_video
from s2_curation.curation_processing import curation_processing
from s3_obj_detect_crop.detect_reframe import generate_transitions_json, process_videos
from s4_add_subtitle.sub import process_subtitles
from s5_bg_music_adder.bg_adder import process_background_music
from helpers.cleaner import clean_files
from helpers.aws_uploader import upload_to_s3

app = FastAPI()


# Define request body model
class VideoRequest(BaseModel):
    youtubeLink: str
    duration: int
    numOfshorts: int
    reframe: bool
    language: str
    fontStyle: str
    captions: bool
    bgm: Optional[str] = None  # Optional field, defaults to None


@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        # Validate duration
        if request.duration not in [45, 60]:
            raise HTTPException(
                status_code=400, detail="Duration must be either 45 or 60 seconds"
            )

        if request.numOfshorts > 5:
            raise HTTPException(
                status_code=400, detail="Number of shorts cannot exceed 5"
            )

        # Print request
        print("----------------------------------")
        print(f"Request: {request}")
        print("----------------------------------")

        # # Step 1: Download video
        # print("----------------------------------")
        # print("STEP 1: Downloading video...")
        # try:
        #     download_video(request.youtubeLink)
        # except Exception as download_error:
        #     if "Video exceeds 45-minute time limit" in str(download_error):
        #         raise HTTPException(
        #             status_code=400, detail="Video duration exceeds 45-minute limit"
        #         )
        #     raise  # Re-raise other download errors
        # print("----------------------------------")

        # Step 2: Process curation
        print("----------------------------------")
        print("STEP 2: Processing curation...")
        curation_processing(duration=request.duration, numOfShorts=request.numOfshorts)
        print("----------------------------------")

        # Step 3: Generate transitions and process videos (only if reframe=True)
        if request.reframe:
            print("----------------------------------")
            print("STEP 3-a: Generating transition data...")
            generate_transitions_json()
            print("----------------------------------")

            print("----------------------------------")
            print("STEP 3-b: Processing videos...")
            process_videos()
            print("----------------------------------")
        else:
            print("----------------------------------")
            print("Skipping STEP 3 (reframe=False)")
            print("----------------------------------")

        # Step 4: Process subtitles (only if captions=True)
        if request.captions:
            print("----------------------------------")
            print(f"STEP 4: Processing subtitles with font: {request.fontStyle}")
            process_subtitles(reframe=request.reframe, style_key=request.fontStyle)
            print("----------------------------------")
        else:
            print("----------------------------------")
            print("Skipping STEP 4 (captions=False)")
            print("----------------------------------")

        # Step 5: Add background music (only if bgm is provided and not empty)
        if request.bgm and request.bgm.strip():  # Check if bgm is not None or empty
            print("----------------------------------")
            print(f"STEP 5: Adding background music: {request.bgm}")
            process_background_music(request.bgm)
            print("----------------------------------")
        else:
            print("----------------------------------")
            print("Skipping STEP 5 (bgm not provided)")
            print("----------------------------------")

        # Step 6: Upload to S3 and collect presigned URLs
        print("----------------------------------")
        print("STEP 6: Uploading to S3...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Determine the source folder based on bgm
        if request.bgm and request.bgm.strip():
            video_dir = os.path.join(script_dir, "downloads/final_video_with_bg")
            file_pattern = "final_vid_with_bg_*.mp4"
        else:
            video_dir = os.path.join(script_dir, "downloads")
            file_pattern = "final_video_*.mp4"

        # Find all video files
        video_files = sorted(glob.glob(os.path.join(video_dir, file_pattern)))
        if not video_files:
            raise HTTPException(
                status_code=500, detail=f"No processed video files found in {video_dir}"
            )

        # Upload to S3 and collect presigned URLs
        video_urls = []
        for video_file in video_files:
            presigned_url = upload_to_s3(video_file, request.duration)
            if "ERROR" in presigned_url:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload {video_file}: {presigned_url}",
                )
            video_urls.append(presigned_url)
            print(f"Uploaded: {video_file} -> {presigned_url}")

        # Step 7: Clean up (optional, uncomment if needed)
        # print("----------------------------------")
        # print("STEP 7: Cleaning up files...")
        # clean_files()
        # print("----------------------------------")

        print("----------------------------------")
        print("Processing complete!")
        print("----------------------------------")

        return {"status": "success", "video_files": video_urls}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Optional: Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Video Processing API"}
