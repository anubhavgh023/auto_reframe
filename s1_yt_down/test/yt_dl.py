import yt_dlp
import os


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        yt_dlp.utils.get_executable_path("ffmpeg")
        return True
    except:
        return False


def get_best_format(ffmpeg_available):
    if ffmpeg_available:
        return "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best"
    else:
        return "best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/best[height<=360][ext=mp4]/best"


def download_video(url, output_path="../downloads"):
    """
    Download a YouTube video with default quality of 720p (fallback to 480p or 360p).

    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the downloaded video
    """
    try:
        # Check if FFmpeg is available
        ffmpeg_available = check_ffmpeg()

        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Configure yt-dlp options
        ydl_opts = {
            "outtmpl": os.path.join(output_path, "%(input_video.%(ext)s"),
            "format": get_best_format(ffmpeg_available),
            "quiet": True,  # Suppress yt-dlp output
        }

        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print("Download completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Example usage
    video_url = input("Enter YouTube video URL: ")
    download_video(video_url)