import cv2
import mediapipe as mp
import os
import ffmpeg

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

# Paths for input and output video
input_video_path = 'videos/input_video.mp4'
output_video_path = 'videos/output_video.mp4'

# Desired output dimensions - mobile-friendly (9:16 aspect ratio) 
output_width = 720  
output_height = 1280  

def detect_faces(frame):
    """Detect faces in a given video frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe works with RGB images
    results = mp_face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            return bbox
    return None

def reframe_video(input_path, output_path):
    """Read the video, detect faces, and reframe to 9:16 aspect ratio."""
    # Open the video file
    video_capture = cv2.VideoCapture(input_path)
    
    # Get original video dimensions
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Prepare FFmpeg writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    temp_output_path = 'temp_video.mp4'  # Temporary video file without audio
    output_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect the face in the current frame
        bbox = detect_faces(frame)
        if bbox:
            # Convert bounding box from relative coordinates to pixel coordinates
            x1 = int(bbox.xmin * original_width)
            y1 = int(bbox.ymin * original_height)
            width = int(bbox.width * original_width)
            height = int(bbox.height * original_height)

            # Calculate the center of the face
            x_center = x1 + width // 2
            y_center = y1 + height // 2

            # Calculate the cropping region for 9:16 aspect ratio centered on the face
            crop_x1 = max(0, x_center - output_width // 2)
            crop_y1 = max(0, y_center - output_height // 2)

            # Ensure the crop fits within the frame dimensions
            crop_x2 = min(original_width, crop_x1 + output_width)
            crop_y2 = min(original_height, crop_y1 + output_height)

            # Crop the frame
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Resize the cropped frame to the desired output size
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height))

            # Write the resized frame to the output video
            output_video.write(resized_frame)

    # Release the video objects
    video_capture.release()
    output_video.release()

    print(f"Reframed video saved to: {output_path}")

if __name__ == '__main__':
    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
    else:
        reframe_video(input_video_path, output_video_path)
