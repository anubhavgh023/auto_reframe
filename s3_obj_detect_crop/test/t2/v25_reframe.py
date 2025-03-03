import cv2
import numpy as np
import mediapipe as mp
import os
from moviepy.editor import VideoFileClip
import logging
from collections import deque

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CropWindow:
    def __init__(self, smoothing_frames=30):
        self.positions = deque(maxlen=smoothing_frames)
        self.current_x = None

    def update(self, new_x, frame_width, crop_width):
        # Add new position to history
        self.positions.append(new_x)

        # Calculate smoothed position
        if self.current_x is None:
            self.current_x = new_x
        else:
            # Get median position from history for stability
            target_x = int(np.median(self.positions))
            # Smooth movement (lerp)
            self.current_x = int(0.8 * self.current_x + 0.2 * target_x)

        # Ensure within frame bounds
        self.current_x = max(0, min(self.current_x, frame_width - crop_width))
        return self.current_x


class PodcastToShortsConverter:
    def __init__(self, input_video_path, output_dir):
        """Initialize the converter with improved performance settings"""
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        self.input_video_path = input_video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # MediaPipe face detection setup
        self.mp_face_detection = mp.solutions.face_detection

        # Video properties
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target dimensions
        self.target_width = 1080
        self.target_height = 1920

        # Processing optimization parameters
        self.detection_interval = 5  # Only detect faces every N frames
        self.processing_scale = 0.5  # Process smaller frames for detection

        # Initialize crop window smoother
        self.crop_window = CropWindow(
            smoothing_frames=int(self.fps)
        )  # 1 second of frames

        logger.info(f"Initialized converter for {input_video_path}")

    def detect_faces(self, frame):
        """Optimized face detection"""
        # Scale down frame for faster processing
        h, w = frame.shape[:2]
        small_frame = cv2.resize(
            frame, (int(w * self.processing_scale), int(h * self.processing_scale))
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(rgb_small_frame)
            faces = []

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)  # Scale back to original size
                    y = int(bbox.ymin * h)
                    w = int(bbox.width * w)
                    h = int(bbox.height * h)
                    faces.append((x, y, w, h))

            return faces

    def get_crop_region(self, faces, prev_crop_x=None):
        """Get stable crop region based on face positions"""
        if not faces:
            if prev_crop_x is not None:
                return prev_crop_x
            return self.width // 2 - (self.height * 9 // 32)

        # Calculate center of all faces
        face_centers = [(f[0] + f[2] // 2) for f in faces]
        center_x = int(np.mean(face_centers))

        # Calculate crop width based on aspect ratio
        crop_width = int(self.height * 9 / 16)
        target_x = center_x - (crop_width // 2)

        # Use smoother to stabilize movement
        smooth_x = self.crop_window.update(target_x, self.width, crop_width)
        return smooth_x

    def process_video(self):
        """Process video with improved performance and stability"""
        temp_output = os.path.join(
            self.output_dir, f"temp_{os.path.basename(self.input_video_path)}"
        )
        final_output = os.path.join(
            self.output_dir, f"vertical_{os.path.basename(self.input_video_path)}"
        )

        try:
            # First pass: Process frames
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                temp_output, fourcc, self.fps, (self.target_width, self.target_height)
            )

            frame_count = 0
            last_faces = None
            last_crop_x = None

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Only detect faces periodically to improve performance
                if frame_count % self.detection_interval == 0:
                    faces = self.detect_faces(frame)
                    last_faces = faces

                # Get crop region
                crop_x = self.get_crop_region(last_faces, last_crop_x)
                last_crop_x = crop_x

                # Crop and resize
                crop_width = int(self.height * 9 / 16)
                cropped = frame[:, crop_x : crop_x + crop_width]
                vertical_frame = cv2.resize(
                    cropped,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                out.write(vertical_frame)
                frame_count += 1

                if frame_count % (self.fps * 5) == 0:  # Log every 5 seconds
                    logger.info(f"Processed {frame_count/self.fps:.1f} seconds")

            self.cap.release()
            out.release()

            # Second pass: Add audio
            logger.info("Processing audio...")
            original_clip = VideoFileClip(self.input_video_path)
            processed_clip = VideoFileClip(temp_output)

            if original_clip.audio is not None:
                final_clip = processed_clip.set_audio(original_clip.audio)
            else:
                final_clip = processed_clip

            final_clip.write_videofile(
                final_output,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                threads=4,  # Use multiple threads for faster encoding
            )

            # Cleanup
            original_clip.close()
            processed_clip.close()
            final_clip.close()

            if os.path.exists(temp_output):
                os.remove(temp_output)

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