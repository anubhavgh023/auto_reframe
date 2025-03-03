import cv2
import numpy as np
import mediapipe as mp
import os
from moviepy.editor import VideoFileClip
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PodcastToShortsConverter:
    def __init__(self, input_video_path, output_dir):
        """
        Initialize the converter with input video and output directory
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        self.input_video_path = input_video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initializing converter for {input_video_path}")

        # MediaPipe face detection setup
        self.mp_face_detection = mp.solutions.face_detection

        # Open video and verify it's readable
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps}fps")

        # Target dimensions (9:16 aspect ratio)
        self.target_width = 1080
        self.target_height = 1920

    def detect_faces(self, frame):
        """Detect faces in a frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(rgb_frame)
            faces = []

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame.shape[1])
                    y = int(bbox.ymin * frame.shape[0])
                    w = int(bbox.width * frame.shape[1])
                    h = int(bbox.height * frame.shape[0])
                    faces.append((x, y, w, h))

            return faces

    def crop_to_vertical(self, frame, faces):
        """Crop frame to vertical format focusing on faces"""
        if not faces:
            # If no faces, center crop
            center = self.width // 2
            crop_width = int(self.height * 9 / 16)
            start_x = center - (crop_width // 2)
            start_x = max(0, min(start_x, self.width - crop_width))
            cropped = frame[:, start_x : start_x + crop_width]
        else:
            # Crop around faces
            min_x = min(face[0] for face in faces)
            max_x = max(face[0] + face[2] for face in faces)

            center_x = (min_x + max_x) // 2
            crop_width = int(self.height * 9 / 16)
            start_x = center_x - (crop_width // 2)
            start_x = max(0, min(start_x, self.width - crop_width))

            cropped = frame[:, start_x : start_x + crop_width]

        return cv2.resize(
            cropped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def process_video(self):
        """Process video and create vertical format with audio"""
        temp_output = os.path.join(
            self.output_dir, f"temp_{os.path.basename(self.input_video_path)}"
        )
        final_output = os.path.join(
            self.output_dir, f"vertical_{os.path.basename(self.input_video_path)}"
        )

        logger.info("Starting video processing")

        try:
            # First pass: Process frames and save temporary video
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                temp_output, fourcc, self.fps, (self.target_width, self.target_height)
            )

            if not out.isOpened():
                raise RuntimeError("Failed to create video writer")

            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                faces = self.detect_faces(frame)
                vertical_frame = self.crop_to_vertical(frame, faces)
                out.write(vertical_frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

            self.cap.release()
            out.release()

            if frame_count == 0:
                raise RuntimeError("No frames were processed")

            if not os.path.exists(temp_output):
                raise RuntimeError("Temporary file was not created")

            logger.info("Video processing complete. Adding audio...")

            # Second pass: Add audio using moviepy
            try:
                # Load the original video with audio
                original_clip = VideoFileClip(self.input_video_path)

                # Load the processed video
                processed_clip = VideoFileClip(temp_output)

                if original_clip.audio is not None:
                    # Add the original audio to the processed video
                    final_clip = processed_clip.set_audio(original_clip.audio)
                else:
                    logger.warning("No audio found in original video")
                    final_clip = processed_clip

                # Write the final video with audio
                final_clip.write_videofile(
                    final_output,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                )

                # Clean up
                original_clip.close()
                processed_clip.close()
                final_clip.close()

            except Exception as e:
                raise RuntimeError(f"Error during audio processing: {str(e)}")

            finally:
                # Clean up temporary files
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                if os.path.exists("temp-audio.m4a"):
                    os.remove("temp-audio.m4a")

            logger.info(f"Successfully created: {final_output}")
            return final_output

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise


def convert_podcast_to_shorts(input_dir, output_dir):
    """Convert all videos in input directory to vertical format"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".mp4", ".mov", ".avi")):
            input_path = os.path.join(input_dir, filename)
            logger.info(f"\nProcessing: {filename}")

            try:
                converter = PodcastToShortsConverter(input_path, output_dir)
                output_path = converter.process_video()
                logger.info(f"Successfully processed: {filename} → {output_path}")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")


if __name__ == "__main__":
    input_video_dir = "../2.curation/assets/curated_videos"
    output_video_dir = "../2.curation/assets/processed_shorts"

    try:
        convert_podcast_to_shorts(input_video_dir, output_video_dir)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")