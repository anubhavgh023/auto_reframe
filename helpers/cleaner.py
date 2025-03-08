# import os
# import shutil

# def clean_files():
#     """
#     Cleans up temporary files from the processing pipeline.

#     Deletes files in:
#     - ./downloads (while preserving files in ./downloads/final_video_with_bg)
#     - ./s2_curation/assets
#     """
#     print("Starting cleanup process...")

#     # Clean downloads directory
#     downloads_dir = "./downloads"
#     if os.path.exists(downloads_dir):
#         # Get list of files and directories in downloads
#         items = os.listdir(downloads_dir)

#         for item in items:
#             item_path = os.path.join(downloads_dir, item)

#             # Skip the final_video_with_bg directory
#             if item == "final_video_with_bg":
#                 continue

#             try:
#                 if os.path.isfile(item_path):
#                     os.remove(item_path)
#                     print(f"Deleted file: {item_path}")
#                 elif os.path.isdir(item_path):
#                     shutil.rmtree(item_path)
#                     print(f"Deleted directory: {item_path}")
#             except Exception as e:
#                 print(f"Error deleting {item_path}: {str(e)}")

#     # Clean s2_curation/assets directory
#     curation_assets_dir = "./s2_curation/assets"
#     if os.path.exists(curation_assets_dir):
#         try:
#             # Delete all files in the assets directory
#             for filename in os.listdir(curation_assets_dir):
#                 file_path = os.path.join(curation_assets_dir, filename)
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#                     print(f"Deleted file: {file_path}")
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#                     print(f"Deleted directory: {file_path}")
#             print(f"Cleaned {curation_assets_dir}")
#         except Exception as e:
#             print(f"Error cleaning {curation_assets_dir}: {str(e)}")
#     print("Cleanup completed!")


#----------------------------------------------------#
import os
import shutil

# Global list of files/directories that should not be deleted during cleanup
PRESERVED_FILES = [
    "final_video_with_bg",  # Directory with final videos
    "final_videos",
]


def clean_files():
    """
    Cleans up temporary files from the processing pipeline,
    while preserving specific files defined in PRESERVED_FILES.

    Deletes files in:
    - ./downloads (while preserving files in ./downloads/final_video_with_bg and bg_1.mp3)
    - ./s2_curation/assets
    """
    print("Starting cleanup process...")

    # Clean downloads directory
    downloads_dir = "./downloads"
    if os.path.exists(downloads_dir):
        # Get list of files and directories in downloads
        items = os.listdir(downloads_dir)

        for item in items:
            item_path = os.path.join(downloads_dir, item)

            # Skip files/directories in the preserved list
            if item in PRESERVED_FILES:
                print(f"Preserving: {item_path}")
                continue

            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted directory: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {str(e)}")

    # Clean s2_curation/assets directory
    curation_assets_dir = "./s2_curation/assets"
    if os.path.exists(curation_assets_dir):
        try:
            # Delete all files in the assets directory
            for filename in os.listdir(curation_assets_dir):
                file_path = os.path.join(curation_assets_dir, filename)

                # Skip files in the preserved list (with just filename, no path)
                if filename in PRESERVED_FILES:
                    print(f"Preserving: {file_path}")
                    continue

                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
            print(f"Cleaned {curation_assets_dir}")
        except Exception as e:
            print(f"Error cleaning {curation_assets_dir}: {str(e)}")

    print("Cleanup completed!")


# Example usage
if __name__ == "__main__":
    clean_files()