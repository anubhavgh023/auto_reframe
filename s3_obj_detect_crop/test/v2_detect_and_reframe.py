import cv2
import mediapipe as mp
import os
import ffmpeg

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.7
)

# Paths for input and output video
input_video_path = "videos/input_video.mp4"
output_video_path = "videos/output_video.mp4"  # Final output video with audio included

# Desired output dimensions (9:16 aspect ratio)
output_width = 720
output_height = 1280

# Variables for smoothing (keeps track of previous positions)
prev_x_center = None
prev_y_center = None
smooth_factor = 0.85  # Adjust smoothing factor (between 0 and 1)


def detect_faces(frame):
    """
    Detect faces in a given video frame using MediaPipe.
    Returns the bounding box of the detected face, or None if no face is found.
    """
    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # Convert BGR to RGB for MediaPipe
    results = mp_face_detection.process(rgb_frame)  # Perform face detection

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            return bbox  # Return the bounding box of the first detected face
    return None


def reframe_video(input_path, output_path):
    """
    Reframe the video to 9:16 aspect ratio by detecting faces and cropping the video accordingly.
    Apply smoothing to avoid abrupt movements in the cropped video.
    """
    video_capture = cv2.VideoCapture(input_path)

    # Get the original video properties
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create a temporary output file for the video without audio
    temp_output_path = "videos/temp_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        temp_output_path, fourcc, fps, (output_width, output_height)
    )

    global prev_x_center, prev_y_center  # To track the previous face position

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit loop when the video ends

        # Detect face and get bounding box
        bbox = detect_faces(frame)
        if bbox:
            # Calculate the bounding box in pixel coordinates
            x1 = int(bbox.xmin * original_width)
            y1 = int(bbox.ymin * original_height)
            width = int(bbox.width * original_width)
            height = int(bbox.height * original_height)

            # Get the center of the bounding box
            x_center = x1 + width // 2
            y_center = y1 + height // 2

            # Apply smoothing to the center position
            if prev_x_center is not None and prev_y_center is not None:
                x_center = int(
                    smooth_factor * prev_x_center + (1 - smooth_factor) * x_center
                )
                y_center = int(
                    smooth_factor * prev_y_center + (1 - smooth_factor) * y_center
                )

            # Update previous center for next frame
            prev_x_center, prev_y_center = x_center, y_center

            # Calculate the cropping box for 9:16 aspect ratio
            crop_x1 = max(0, x_center - output_width // 2)
            crop_y1 = max(0, y_center - output_height // 2)
            crop_x2 = min(original_width, crop_x1 + output_width)
            crop_y2 = min(original_height, crop_y1 + output_height)

            # Crop and resize the frame to the desired output size
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height))

            # Write the processed frame to the temporary video
            output_video.write(resized_frame)

    # Release the video objects
    video_capture.release()
    output_video.release()

    # Add the audio from the original video to the processed video
    add_audio(input_path, temp_output_path, output_path)

    # Clean up the temporary video file
    os.remove(temp_output_path)
    print(f"Reframed video with audio saved to: {output_path}")


def add_audio(input_video, processed_video, final_output):
    """
    Use ffmpeg to add audio from the original video to the processed video.
    The video will have the same content, but with the original audio.
    """
    # Use ffmpeg to merge the video with the audio, ensuring we copy the video and re-encode the audio
    ffmpeg.input(processed_video).input(input_video).output(
        final_output, **{"c:v": "copy", "c:a": "aac", "map": "0:v:0", "map": "1:a:0"}
    ).run()


if __name__ == "__main__":
    # Check if the input video file exists
    if not os.path.exists(input_video_path):
        print(f"Input video file not found at {input_video_path}")
    else:
        # Process the video by detecting faces, cropping, and adding audio
        reframe_video(input_video_path, output_video_path)
