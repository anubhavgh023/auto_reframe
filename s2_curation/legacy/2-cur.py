import os
import subprocess
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
INPUT_VIDEO = "downloads/input_video.mp4"
AUDIO_FILE = "downloads/extracted_audio.wav"
OUTPUT_DIR = "downloads"
CURATED_VIDEO = os.path.join(OUTPUT_DIR, "curated_video.mp4")

# Target durations
TARGET_DURATIONS = [45, 60, 90]

def extract_audio(input_video, output_audio):
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", 
            "-i", input_video, 
            "-vn",  # No video
            "-acodec", "pcm_s16le", 
            "-ar", "16000",  # Sample rate
            "-ac", "1",  # Mono
            output_audio
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction error: {e.stderr}")
        raise

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        return transcript.words
    except Exception as e:
        print(f"Transcription error: {e}")
        raise

def find_best_segment(words, target_duration):
    """Find most engaging segment using GPT-4"""
    try:
        # Convert words to transcript text with timestamps
        transcript_text = "\n".join([f"{word.start:.2f}-{word.end:.2f}: {word.word}" for word in words])
        
        # Prompt for segment selection
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"""Analyze the transcript and identify the most engaging {target_duration}-second segment:
                    1. Strong narrative hook
                    2. High emotional or informative value
                    3. Maintain full context
                    Return ONLY JSON: {{"start": start_sec, "end": end_sec}}"""
                },
                {"role": "user", "content": transcript_text}
            ],
            response_format={"type": "json_object"}
        )

        # Parse response
        segment = json.loads(response.choices[0].message.content)
        return [(segment["start"], segment["end"])]
    
    except Exception as e:
        print(f"Segment selection error: {e}")
        # Fallback: first available segment
        return [(words[0].start, words[-1].end)]


def crop_video(input_video, output_video, segments):
    """Crop video using ffmpeg"""
    try:
        # Prepare filter complex command for precise segment extraction
        filter_parts = []
        input_parts = []
        
        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
            input_parts.append(f"[v{i}][a{i}]")
        
        # Concatenate filter
        filter_parts.append(f"{''.join(input_parts)}concat=n={len(segments)}:v=1:a=1[outv][outa]")
        
        # Full FFmpeg command
        cmd = [
            "ffmpeg", 
            "-i", input_video, 
            "-filter_complex", ";".join(filter_parts),
            "-map", "[outv]", 
            "-map", "[outa]", 
            "-c:v", "libx264", 
            "-c:a", "aac", 
            output_video
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_video
    
    except subprocess.CalledProcessError as e:
        print(f"Video cropping error: {e.stderr}")
        raise

def main():
    # Extract audio
    audio_path = extract_audio(INPUT_VIDEO, AUDIO_FILE)
    
    # Transcribe audio
    words = transcribe_audio(audio_path)
    
    # Find best segment
    best_segments = find_best_segment(words, TARGET_DURATIONS[0])
    
    # Crop and save video
    crop_video(INPUT_VIDEO, CURATED_VIDEO, best_segments)
    
    print(f"Curated video saved to: {CURATED_VIDEO}")

if __name__ == "__main__":
    main()
