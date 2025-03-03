import cv2
import numpy as np
import mediapipe as mp
import os
import webrtcvad
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import librosa
import subprocess
import wave
import array
import tempfile
from pathlib import Path


class PodcastToShortsConverter:
    def __init__(self, input_video_path, output_dir):
        """Initialize converter with video path and output directory"""
        self.input_video_path = input_video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

        # Open video and get properties
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Target dimensions for vertical video (9:16 aspect ratio)
        self.target_width = 1080
        self.target_height = 1920

        # Extract audio for speaker detection
        self.temp_dir = Path(tempfile.mkdtemp())
        self._extract_audio()

    def _extract_audio(self):
        """Extract audio from video file for speaker detection"""
        # Extract audio using ffmpeg
        temp_wav = self.temp_dir / "temp_audio.wav"
        cmd = [
            "ffmpeg",
            "-i",
            str(self.input_video_path),
            "-ac",
            "2",
            "-ar",
            "16000",
            "-vn",
            str(temp_wav),
            "-y",
        ]
        subprocess.run(cmd, capture_output=True)

        # Load audio file
        self.audio, self.sample_rate = librosa.load(temp_wav, sr=16000, mono=False)
        temp_wav.unlink()

    def detect_speaker(self, frame_num):
        """Detect which speaker is active in the given frame"""
        # Convert frame number to audio sample position
        start_sample = int(frame_num / self.fps * self.sample_rate)
        window_size = int(0.1 * self.sample_rate)  # 100ms window

        if start_sample + window_size >= self.audio.shape[1]:
            return None

        # Get audio chunk and calculate energy for each channel
        audio_chunk = self.audio[:, start_sample : start_sample + window_size]
        left_energy = np.sum(np.abs(audio_chunk[0]))
        right_energy = np.sum(np.abs(audio_chunk[1]))

        # Determine active speaker based on energy difference
        energy_ratio = 1.2  # Required difference for speaker detection
        if left_energy > right_energy * energy_ratio:
            return 0  # Left speaker
        elif right_energy > left_energy * energy_ratio:
            return 1  # Right speaker
        return None  # No clear speaker

    def detect_faces(self, frame):
        """Detect and return face locations in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        faces = []

        if results.detections:
            frame_height, frame_width = frame.shape[:2]

            # Get all faces and sort by x position
            all_faces = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)
                all_faces.append((x, y, w, h))

            # Sort faces left to right and assign indices
            all_faces.sort(key=lambda f: f[0])
            for idx, face in enumerate(all_faces):
                faces.append((*face, idx))

        return faces

    def crop_to_speaker(self, frame, faces, speaker_idx):
        """Crop frame to focus on the active speaker"""
        if not faces:
            # Default center crop if no faces detected
            crop_width = int(self.frame_height * 9 / 16)
            start_x = (self.frame_width - crop_width) // 2
            crop = frame[:, start_x : start_x + crop_width]
            return cv2.resize(crop, (self.target_width, self.target_height))

        # Find active speaker's face
        speaker_face = None
        for face in faces:
            if face[4] == speaker_idx:
                speaker_face = face
                break

        if speaker_face is None:
            # Use default crop if speaker's face not found
            crop_width = int(self.frame_height * 9 / 16)
            start_x = (self.frame_width - crop_width) // 2
        else:
            # Center crop on speaker's face
            x, y, w, h, _ = speaker_face
            face_center = x + w // 2
            crop_width = int(self.frame_height * 9 / 16)

            # Calculate crop position with padding
            padding = int(crop_width * 0.2)  # 20% padding
            start_x = face_center - (crop_width // 2)
            start_x = max(0, min(start_x, self.frame_width - crop_width))

        # Perform the crop and resize
        crop = frame[:, start_x : start_x + crop_width]
        return cv2.resize(crop, (self.target_width, self.target_height))

    def process_video(self):
        """Process the entire video and create shorts"""
        output_path = self.output_dir / f"{Path(self.input_video_path).stem}_shorts.mp4"

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.target_width, self.target_height)
        )

        # Processing variables
        frame_num = 0
        last_speaker = None
        speaker_switch_frame = 0
        min_switch_frames = int(self.fps * 0.5)  # Minimum 0.5s between switches

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces and active speaker
            faces = self.detect_faces(frame)
            current_speaker = self.detect_speaker(frame_num)

            # Apply hysteresis to speaker switching
            if current_speaker is not None:
                if current_speaker != last_speaker:
                    if frame_num - speaker_switch_frame >= min_switch_frames:
                        last_speaker = current_speaker
                        speaker_switch_frame = frame_num

            # Use the stable speaker detection
            speaker_to_use = (
                current_speaker if current_speaker is not None else last_speaker
            )
            if speaker_to_use is None:
                speaker_to_use = 0  # Default to first speaker if none detected

            # Create vertical frame
            vertical_frame = self.crop_to_speaker(frame, faces, speaker_to_use)
            out.write(vertical_frame)

            # Progress tracking
            frame_num += 1
            if frame_num % 100 == 0:
                progress = (frame_num / self.total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")

        # Cleanup
        self.cap.release()
        out.release()
        self.face_detection.close()

        print(f"Processing complete. Output saved to: {output_path}")
        return output_path


def main():
    input_path = Path("../2.curation/assets/curated_videos")
    output_dir = Path("../2.curation/assets/processed_videos")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_file in input_path.glob("curated_vid_*.mp4"):
        # Get the video number from filename
        video_num = video_file.stem.split("_")[-1]
        output_filename = f"processed_vid_{video_num}.mp4"

        print(f"\nProcessing: {video_file.name}")
        converter = PodcastToShortsConverter(str(video_file), str(output_dir))
        converter.process_video()

        # Rename the output file to match desired format
        old_output = output_dir / f"{video_file.stem}_shorts.mp4"
        new_output = output_dir / output_filename
        if old_output.exists():
            old_output.rename(new_output)
            print(f"Saved as: {output_filename}")


if __name__ == "__main__":
    main()