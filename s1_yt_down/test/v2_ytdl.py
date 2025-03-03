import yt_dlp
import os


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        yt_dlp.utils.get_executable_path("ffmpeg")
        return True
    except:
        return False


def get_format_string(preferred_quality, ffmpeg_available):
    """
    Generate the format string based on preferred quality with fallback options.

    Args:
        preferred_quality (int): Preferred video quality (1080, 720, 480, or 360)
        ffmpeg_available (bool): Whether FFmpeg is available for post-processing

    Returns:
        str: Format string for yt-dlp
    """
    # Define quality tiers from highest to lowest
    quality_tiers = [1080, 720, 480, 360]

    # Find index of preferred quality
    try:
        start_index = quality_tiers.index(preferred_quality)
    except ValueError:
        print(
            f"Invalid quality {preferred_quality}. Defaulting to highest available quality."
        )
        start_index = 0

    # Generate format string with fallback options
    if ffmpeg_available:
        # With FFmpeg, we can merge separate video and audio streams
        format_parts = []
        for quality in quality_tiers[start_index:]:
            format_parts.append(
                f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]"
            )
        format_parts.append("best")  # Final fallback
        return "/".join(format_parts)
    else:
        # Without FFmpeg, we need pre-merged formats
        format_parts = []
        for quality in quality_tiers[start_index:]:
            format_parts.append(f"best[height<={quality}][ext=mp4]")
        format_parts.append("best")  # Final fallback
        return "/".join(format_parts)


def download_video(url, output_path="downloads", preferred_quality=1080):
    try:
        # Check if FFmpeg is available
        ffmpeg_available = check_ffmpeg()
        if not ffmpeg_available:
            print("Warning: FFmpeg not found. Some features may be limited.")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Get format string based on preferred quality
        format_string = get_format_string(preferred_quality, ffmpeg_available)

        # Configure yt-dlp options
        ydl_opts = {
            "format": format_string,
            "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
            "quiet": True,
            "progress_hooks": [progress_hook],
            "postprocessor_hooks": [postprocessor_hook],
        }

        print(f"Attempting to download video at {preferred_quality}p quality...")
        print(
            "Will fall back to next best quality if preferred quality is unavailable."
        )

        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, extract video information
            info = ydl.extract_info(url, download=False)
            available_formats = info.get("formats", [])

            # Print available qualities
            print("\nAvailable qualities:")
            heights = set()
            for f in available_formats:
                if f.get("height"):
                    heights.add(f["height"])
            print(f"Found: {sorted(heights, reverse=True)}p")

            # Proceed with download
            ydl.download([url])

        print("\nDownload completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def progress_hook(d):
    """Hook to track download progress."""
    if d["status"] == "downloading":
        try:
            percent = d["_percent_str"]
            speed = d["_speed_str"]
            print(f"\rDownloading... {percent} at {speed}", end="", flush=True)
        except:
            pass
    elif d["status"] == "finished":
        print("\nDownload finished. Processing...")


def postprocessor_hook(d):
    """Hook to track post-processing progress."""
    if d["status"] == "started":
        print(f"Post-processing: {d['postprocessor']}")
    elif d["status"] == "finished":
        print(f"Finished {d['postprocessor']}")


def main():
    video_url = "https://www.youtube.com/watch?v=8OHYynw7Yh4"
    download_video(video_url, preferred_quality=1280)


if __name__ == "__main__":
    main()
