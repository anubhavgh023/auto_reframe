import yt_dlp
import os
import sys
from datetime import datetime
import subprocess


def check_ffmpeg():
    """
    Check if FFmpeg is installed and accessible.
    Returns True if FFmpeg is available, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def format_size(bytes):
    """
    Convert bytes to human readable format, making file sizes easier to understand.
    Scales from bytes up to gigabytes automatically.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} GB"


def progress_hook(d):
    """
    Display download progress with detailed information about speed and time remaining.
    Provides real-time feedback during the download process.
    """
    if d["status"] == "downloading":
        downloaded = d.get("downloaded_bytes", 0)
        total = d.get("total_bytes", 0) or d.get("total_bytes_estimate", 0)

        if total:
            percentage = (downloaded / total) * 100
            speed = d.get("speed", 0)
            speed_str = format_size(speed) + "/s" if speed else "N/A"

            eta = d.get("eta", None)
            eta_str = (
                str(datetime.fromtimestamp(eta).strftime("%M:%S")) if eta else "N/A"
            )

            progress = (
                f"\rProgress: {percentage:.1f}% | Speed: {speed_str} | ETA: {eta_str}"
            )
            sys.stdout.write(progress)
            sys.stdout.flush()


def get_format_for_quality(preferred_qualities, ffmpeg_available):
    """
    Create a format string for yt-dlp that tries each quality in order.

    Args:
        preferred_qualities (list): List of heights to try in order of preference
        ffmpeg_available (bool): Whether FFmpeg is available for separate streams
    """
    if ffmpeg_available:
        # Build a format string that tries each quality in order with FFmpeg
        format_parts = []
        for height in preferred_qualities:
            format_parts.append(
                f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]"
            )

        # Add fallbacks
        format_parts.extend(
            [f"best[height<={height}][ext=mp4]" for height in preferred_qualities]
        )
        format_parts.append("best")

        return "/".join(format_parts)
    else:
        # Build a format string for merged formats
        format_parts = []
        for height in preferred_qualities:
            format_parts.append(f"best[height<={height}][ext=mp4]")

        format_parts.append("best[ext=mp4]")
        format_parts.append("best")

        return "/".join(format_parts)


# def download_video(url, output_path="downloads"):
#     """
#     Download a YouTube video with quality priority: 1080p -> 720p -> 480p.
#     Save as input_video.mp4 in the downloads folder.

#     Args:
#         url (str): YouTube video URL
#         output_path (str): Directory to save the downloaded video
#     """
#     try:
#         # Check if FFmpeg is available
#         ffmpeg_available = check_ffmpeg()
#         if not ffmpeg_available:
#             print(
#                 "\nNotice: FFmpeg is not installed. Some high-quality options may be limited."
#             )
#             print(
#                 "To enable all quality options, please install FFmpeg and add it to your system PATH."
#             )

#         # Create output directory if it doesn't exist
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)

#         # Preferred qualities in order (1080p, 720p, 480p)
#         preferred_qualities = [1080, 720, 480]

#         # Fixed output filename
#         output_file = os.path.join(output_path, "input_video.mp4")

#         # Configure yt-dlp options
#         ydl_opts = {
#             "progress_hooks": [progress_hook],
#             "outtmpl": output_file,
#             "format": get_format_for_quality(preferred_qualities, ffmpeg_available),
#             "merge_output_format": "mp4",
#             "verbose": False,
#         }

#         print("Fetching video information...")

#         # Create a yt-dlp object and get video info
#         with yt_dlp.YoutubeDL({"verbose": False}) as ydl:
#             # Get video information
#             info = ydl.extract_info(url, download=False)

#             # Display video information
#             print(f"\nVideo Title: {info.get('title', 'Unknown')}")
#             duration = int(info.get("duration", 0))
#             print(f"Duration: {duration // 60}:{duration % 60:02d}")


#         print("\nDownloading video with priority: 1080p -> 720p -> 480p")
#         print(f"Output file: {output_file}")

#         # Download the video
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])

#         print("\nDownload completed successfully!")

#     except Exception as e:
#         print(f"\nAn error occurred: {str(e)}")
#         print("\nTroubleshooting tips:")
#         print("1. Check your internet connection")
#         print("2. Verify the video URL is correct and accessible")
#         print("3. Try updating yt-dlp: `pip install --upgrade yt-dlp`")
#         print("4. Make sure the video isn't private or age-restricted")
#         if not ffmpeg_available:
#             print("5. For better quality options, install FFmpeg")


# if __name__ == "__main__":
#     # Get video URL from user
#     video_url = input("Enter YouTube video URL: ")

#     # Download the video with default settings
#     download_video(video_url)


# -----------------------
import os
import re
from urllib.parse import urlparse
import yt_dlp


def download_video(url, output_path="downloads"):
    """
    Download a YouTube video with quality priority: 1080p -> 720p -> 480p.
    Save as input_video.mp4 in the downloads folder.

    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the downloaded video

    Raises:
        ValueError: If URL is invalid or not from YouTube
        Exception: If download fails or video exceeds time limit
    """
    try:
        # Basic URL validation
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")

        # Check if it's a YouTube URL
        youtube_regex = (
            r"(https?://)?(www\.)?"
            r"(youtube\.com|youtu\.be)/"
            r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
        )
        if not re.match(youtube_regex, url):
            raise ValueError("Invalid YouTube URL provided")

        # Check if FFmpeg is available
        ffmpeg_available = check_ffmpeg()
        if not ffmpeg_available:
            print(
                "\nNotice: FFmpeg is not installed. Some high-quality options may be limited."
            )
            print(
                "To enable all quality options, please install FFmpeg and add it to your system PATH."
            )

        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except Exception as dir_error:
                raise Exception(f"Failed to create output directory: {str(dir_error)}")

        # Preferred qualities in order (1080p, 720p, 480p)
        preferred_qualities = [1080, 720, 480]

        # Fixed output filename
        output_file = os.path.join(output_path, "input_video.mp4")

        # Configure yt-dlp options
        ydl_opts = {
            "progress_hooks": [progress_hook],
            "outtmpl": output_file,
            "format": get_format_for_quality(preferred_qualities, ffmpeg_available),
            "merge_output_format": "mp4",
            "verbose": False,
        }

        print("Fetching video information...")

        # Get video info and check duration
        with yt_dlp.YoutubeDL({"verbose": False}) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise Exception("Could not extract video information")

            # Display video information
            print(f"\nVideo Title: {info.get('title', 'Unknown')}")
            duration = int(info.get("duration", 0))
            print(f"Duration: {duration // 60}:{duration % 60:02d}")

            # Check if duration exceeds 45 minutes (2700 seconds)
            if duration > 2700:
                raise Exception("Video exceeds 45-minute time limit")

        print("\nDownloading video with priority: 1080p -> 720p -> 480p")
        print(f"Output file: {output_file}")

        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Verify download
        if not os.path.exists(output_file):
            raise Exception("Download completed but output file not found")

        print("\nDownload completed successfully!")
        return output_file

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the video URL is correct and accessible")
        print("3. Try updating yt-dlp: `pip install --upgrade yt-dlp`")
        print("4. Make sure the video isn't private or age-restricted")
        if not ffmpeg_available:
            print("5. For better quality options, install FFmpeg")
        raise  # Re-raise for caller to handle
