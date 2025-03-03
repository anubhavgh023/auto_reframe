import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from collections import deque, defaultdict
import math
from typing import Dict, List, Tuple

# Initialize MediaPipe Face Detection with better performance
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Mouth landmarks indices (upper and lower lip)
MOUTH_LANDMARKS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 321, 324, 375]

# Paths setup
BASE_PATH = "../downloads"
INPUT_VIDEO = f"{BASE_PATH}/curated_videos/curated_video_01.mp4"
TEMP_FACES_DIR = f"{BASE_PATH}/temp_framed"
FINAL_OUTPUT = f"{BASE_PATH}/final_conversation.mp4"

# Create necessary directories
os.makedirs(TEMP_FACES_DIR, exist_ok=True)

# Video dimensions
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
SPLIT_HEIGHT = OUTPUT_HEIGHT // 2

class FaceTracker:
    def __init__(self, face_id: int, window_size: int = 45):
        self.face_id = face_id
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.width_history = deque(maxlen=window_size)
        self.height_history = deque(maxlen=window_size)
        self.prev_crop_box = None
        self.confidence_history = deque(maxlen=window_size)
        
    def update(self, bbox, confidence: float, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        if bbox is None:
            return self.prev_crop_box if self.prev_crop_box is not None else self._get_default_crop_box(frame_shape)
        
        original_height, original_width = frame_shape[:2]
        
        # Calculate face center and dimensions
        x_center = int(bbox.xmin * original_width + (bbox.width * original_width) / 2)
        y_center = int(bbox.ymin * original_height + (bbox.height * original_height) / 2)
        face_width = int(bbox.width * original_width)
        face_height = int(bbox.height * original_height)
        
        # Update histories
        self.x_history.append(x_center)
        self.y_history.append(y_center)
        self.width_history.append(face_width)
        self.height_history.append(face_height)
        self.confidence_history.append(confidence)
        
        # Calculate crop box with stabilization
        crop_box = self._calculate_stable_crop_box(original_width, original_height)
        
        # Smooth transition
        if self.prev_crop_box is not None:
            crop_box = self._smooth_crop_box(self.prev_crop_box, crop_box)
        
        self.prev_crop_box = crop_box
        return crop_box

    def _calculate_stable_crop_box(self, original_width: int, original_height: int) -> Tuple[int, int, int, int]:
        x_center = int(np.mean(self.x_history))
        y_center = int(np.mean(self.y_history))
        
        # Calculate crop dimensions maintaining aspect ratio
        crop_height = original_height
        crop_width = int(crop_height * (OUTPUT_WIDTH / OUTPUT_HEIGHT))
        
        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / (OUTPUT_WIDTH / OUTPUT_HEIGHT))
            
        x1 = max(0, min(x_center - crop_width // 2, original_width - crop_width))
        y1 = max(0, min(y_center - crop_height // 2, original_height - crop_height))
        
        return (x1, y1, crop_width, crop_height)

    def _smooth_crop_box(self, prev_box: Tuple[int, int, int, int], current_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        smooth_factor = 0.8
        return tuple(int(prev * smooth_factor + current * (1 - smooth_factor)) for prev, current in zip(prev_box, current_box))

    def _get_default_crop_box(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        original_height, original_width = frame_shape[:2]
        crop_height = original_height
        crop_width = int(crop_height * (OUTPUT_WIDTH / OUTPUT_HEIGHT))
        
        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / (OUTPUT_WIDTH / OUTPUT_HEIGHT))
            
        x1 = (original_width - crop_width) // 2
        y1 = (original_height - crop_height) // 2
        return (x1, y1, crop_width, crop_height)

class SpeakerDetector:
    def __init__(self):
        self.prev_landmarks = None
        self.movement_threshold = 0.02
        self.speaking_threshold = 0.5
        self.movement_history = defaultdict(lambda: deque(maxlen=15))
        
    def detect_speaker(self, face_landmarks_list, frame_shape: Tuple[int, int, int]) -> Dict[int, float]:
        height, width = frame_shape[:2]
        current_movements = {}
        
        for face_idx, landmarks in enumerate(face_landmarks_list):
            # Extract mouth landmarks
            mouth_points = [landmarks.landmark[i] for i in MOUTH_LANDMARKS]
            
            if self.prev_landmarks is not None and face_idx in self.prev_landmarks:
                movement = self._calculate_movement(
                    self.prev_landmarks[face_idx],
                    mouth_points,
                    width,
                    height
                )
                self.movement_history[face_idx].append(movement)
                current_movements[face_idx] = np.mean(self.movement_history[face_idx])
            
        # Update previous landmarks
        self.prev_landmarks = {
            face_idx: [landmarks.landmark[i] for i in MOUTH_LANDMARKS]
            for face_idx, landmarks in enumerate(face_landmarks_list)
        }
            
        return current_movements

    def _calculate_movement(self, prev_landmarks: List, current_landmarks: List, width: int, height: int) -> float:
        movements = []
        for p1, p2 in zip(prev_landmarks, current_landmarks):
            dx = (p2.x - p1.x) * width
            dy = (p2.y - p1.y) * height
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        return np.mean(movements)

class VideoProcessor:
    def __init__(self):
        self.face_trackers = {}
        self.speaker_detector = SpeakerDetector()
        
    def process_video(self, input_path: str):
        print("Starting video processing...")
        # First pass: Detect and track faces
        print("Phase 1: Detecting face segments...")
        face_segments = self._detect_face_segments(input_path)
        
        # Second pass: Extract individual face videos
        print("Phase 2: Extracting individual face videos...")
        face_videos = self._extract_face_videos(input_path, face_segments)
        
        # Third pass: Create conversation layout
        print("Phase 3: Creating conversation layout...")
        self._create_conversation_video(face_videos, input_path)
        print("Processing complete!")

    def _detect_face_segments(self, video_path: str) -> Dict[int, List[dict]]:
        cap = cv2.VideoCapture(video_path)
        face_segments = defaultdict(list)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces and speaking status
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                speaking_scores = self.speaker_detector.detect_speaker(
                    face_results.multi_face_landmarks,
                    frame.shape
                )
                
                for face_idx, score in speaking_scores.items():
                    face_segments[face_idx].append({
                        'frame': frame_idx,
                        'speaking': score > self.speaker_detector.speaking_threshold
                    })
            
            if frame_idx % 30 == 0:
                print(f"Detecting faces: {(frame_idx/total_frames)*100:.1f}% complete")
            
            frame_idx += 1
            
        cap.release()
        return face_segments

    def _extract_face_videos(self, video_path: str, face_segments: Dict[int, List[dict]]) -> Dict[int, str]:
        face_videos = {}
        
        for face_idx in face_segments.keys():
            output_path = f"{TEMP_FACES_DIR}/face_{face_idx}.mp4"
            tracker = FaceTracker(face_idx)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            )
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx in [segment['frame'] for segment in face_segments[face_idx]]:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections and len(results.detections) > face_idx:
                        face_bbox = results.detections[face_idx].location_data.relative_bounding_box
                        crop_box = tracker.update(face_bbox, results.detections[face_idx].score, frame.shape)
                        
                        x1, y1, crop_w, crop_h = crop_box
                        cropped = frame[int(y1):int(y1 + crop_h), int(x1):int(x1 + crop_w)]
                        resized = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                        
                        out.write(resized)
                
                if frame_idx % 30 == 0:
                    print(f"Extracting face {face_idx}: {(frame_idx/total_frames)*100:.1f}% complete")
                
                frame_idx += 1
            
            cap.release()
            out.release()
            face_videos[face_idx] = output_path
            
        return face_videos

    def _create_conversation_video(self, face_videos: Dict[int, str], original_video: str):
        print("Extracting audio...")
        # Extract audio from original video
        audio_path = f"{TEMP_FACES_DIR}/audio.wav"
        subprocess.run([
            'ffmpeg', '-i', original_video,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            audio_path
        ])
        
        print("Creating conversation layout...")
        # Create conversation layout video
        temp_output = f"{TEMP_FACES_DIR}/temp_output.mp4"
        
        caps = {idx: cv2.VideoCapture(path) for idx, path in face_videos.items()}
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(
            temp_output,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        )
        
        frame_idx = 0
        while True:
            frames = {}
            for idx, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    break
                frames[idx] = frame
                
            if not frames:
                break
                
            speaking_face = self._determine_speaking_face(frames)
            
            if len(frames) > 2:
                # If more than 2 faces, show speaking face full screen
                out.write(frames[speaking_face])
            else:
                # Create split screen with speaking face on top
                combined = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
                combined[:SPLIT_HEIGHT] = cv2.resize(frames[speaking_face], (OUTPUT_WIDTH, SPLIT_HEIGHT))
                
                # Add listening face below
                listening_face = next(idx for idx in frames.keys() if idx != speaking_face)
                combined[SPLIT_HEIGHT:] = cv2.resize(frames[listening_face], (OUTPUT_WIDTH, SPLIT_HEIGHT))
                
                out.write(combined)
            
            if frame_idx % 30 == 0:
                print(f"Creating layout: {(frame_idx/total_frames)*100:.1f}% complete")
            frame_idx += 1
        
        # Clean up video captures
        for cap in caps.values():
            cap.release()
        out.release()
        
        print("Combining video with audio...")
        # Combine with audio
        subprocess.run([
            'ffmpeg', '-i', temp_output,
            '-i', audio_path,
            '-c:v', 'libx264', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            FINAL_OUTPUT
        ])
        
        print("Cleaning up temporary files...")
        # Cleanup temporary files
        os.remove(temp_output)
        os.remove(audio_path)
        for video_path in face_videos.values():
            os.remove(video_path)

    def _determine_speaking_face(self, frames: Dict[int, np.ndarray]) -> int:
        # This is a placeholder - in a real implementation, you'd want to use
        # audio analysis to determine who is speaking
        return list(frames.keys())[0]

if __name__ == "__main__" :
    processor = VideoProcessor()
    processor.process_video(INPUT_VIDEO)
