import cv2
import numpy as np
import mediapipe as mp
import os

class PodcastToShortsConverter:
    def __init__(self, input_video_path, output_dir):
        """
        Initialize the converter with input video and output directory
        
        Args:
            input_video_path (str): Path to the input horizontal video
            output_dir (str): Directory to save processed vertical videos
        """
        self.input_video_path = input_video_path
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # MediaPipe face detection setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Video capture and properties
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 9:16 target dimensions
        self.target_width = 1080
        self.target_height = 1920
        
    def detect_faces(self, frame):
        """
        Detect faces in a frame using MediaPipe
        
        Args:
            frame (np.ndarray): Input video frame
        
        Returns:
            list: List of detected face locations
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:
            
            results = face_detection.process(rgb_frame)
            faces = []
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame.shape[1])
                    y = int(bbox.ymin * frame.shape[0])
                    w = int(bbox.width * frame.shape[1])
                    h = int(bbox.height * frame.shape[0])
                    
                    faces.append((x, y, w, h))
            
            return faces
    
    def crop_to_vertical(self, frame, face_rect):
        """
        Crop frame to 9:16 vertical format centered on face while preserving aspect ratio
        
        Args:
            frame (np.ndarray): Input frame
            face_rect (tuple): Face rectangle (x, y, w, h)
        
        Returns:
            np.ndarray: Cropped vertical frame
        """
        x, y, w, h = face_rect
        
        # Center of face
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Target aspect ratio
        target_aspect = 9 / 16
        
        # Compute crop width and height maintaining original aspect ratio
        crop_height = self.height
        crop_width = int(crop_height * target_aspect)
        
        # Adjust crop region to center on face
        start_x = max(0, center_x - crop_width // 2)
        start_y = 0  # Keep full vertical height
        
        # Ensure crop stays within frame
        if start_x + crop_width > self.width:
            start_x = self.width - crop_width
        
        # Crop
        cropped = frame[
            start_y:start_y+crop_height, 
            start_x:start_x+crop_width
        ]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (self.target_width, self.target_height))
        return resized
    
    def process_video(self, min_face_duration=0.5):
        """
        Process video, detect faces, and create vertical shorts
        
        Args:
            min_face_duration (float): Minimum duration (seconds) to track a face
        
        Returns:
            str: Path to output video
        """
        # Prepare video writer
        output_filename = os.path.basename(self.input_video_path).split('.')[0] + '_shorts.mp4'
        output_path = os.path.join(self.output_dir, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                               (self.target_width, self.target_height))
        
        # Face tracking variables
        current_face = None
        face_start_time = 0
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            faces = self.detect_faces(frame)
            current_time = frame_count / self.fps
            
            if faces:
                # If multiple faces, choose the largest
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                
                if current_face is None:
                    # New face detected
                    current_face = largest_face
                    face_start_time = current_time
                
                elif self._is_same_face(current_face, largest_face):
                    # Continue tracking same face
                    current_face = largest_face
                
                else:
                    # Face changed, check if previous face was shown long enough
                    if current_time - face_start_time >= min_face_duration:
                        # Write previous face frames
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._write_face_segment(out, current_face)
                    
                    # Reset for new face
                    current_face = largest_face
                    face_start_time = current_time
            
            frame_count += 1
        
        # Close video objects
        self.cap.release()
        out.release()
        
        return output_path
    
    def _is_same_face(self, face1, face2, iou_threshold=0.5):
        """
        Check if two faces are likely the same based on IoU
        
        Args:
            face1 (tuple): First face rectangle
            face2 (tuple): Second face rectangle
            iou_threshold (float): Intersection over Union threshold
        
        Returns:
            bool: Whether faces are considered the same
        """
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        # Compute intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Compute intersection area
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        
        # Compute union area
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        # Compute IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > iou_threshold
    
    def _write_face_segment(self, video_writer, face_rect):
        """
        Write a segment of video focused on a specific face
        
        Args:
            video_writer (cv2.VideoWriter): Output video writer
            face_rect (tuple): Face rectangle to focus on
        """
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Crop and write vertical frame
            vertical_frame = self.crop_to_vertical(frame, face_rect)
            video_writer.write(vertical_frame)

def convert_podcast_to_shorts(input_dir, output_dir):
    """
    Convert all podcast videos in input directory to vertical shorts
    
    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save processed videos
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_dir, filename)
            converter = PodcastToShortsConverter(input_path, output_dir)
            output_path = converter.process_video()
            print(f"Processed {filename} → {output_path}")

# Example usage
input_video_dir = '../2.curation/assets/curated_videos'
output_video_dir = '../2.curation/assets/processed_shorts'
convert_podcast_to_shorts(input_video_dir, output_video_dir)