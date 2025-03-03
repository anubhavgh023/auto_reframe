import os
import json
import re

# Path to the transcript chunks directory
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path
sliding_transcript_dir = os.path.join(script_dir, "assets/audio/sliding_transcript_chunks")
score_dir = os.path.join(script_dir, "assets/scores")


# SCORE ANALYZER RANKER
def rank_chunk_scores(directory=score_dir, max_size=5):
    """
    Read JSON files in the specified directory, extract TotalScore,
    and create a ranking dictionary sorted in descending order.

    Args:
        directory (str): Path to the directory containing JSON files
        max_size (int): Maximum number of items to keep in the ranking

    Returns:
        dict: A dictionary with top N filenames as keys and their TotalScores as values,
              sorted in descending order
    """
    # Initialize an empty dictionary to store scores
    score_ranking = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JSON file
        if filename.startswith("chunk_score_") and filename.endswith(".json"):
            # Construct full file path
            file_path = os.path.join(directory, filename)

            # Read and parse the JSON file
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)

                    # Extract TotalScore
                    total_score = data.get("TotalScore", 0)

                    # Add to ranking dictionary
                    score_ranking[filename] = total_score

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error processing {filename}: {e}")

    # Sort the dictionary by values in descending order
    sorted_ranking = dict(
        sorted(score_ranking.items(), key=lambda item: item[1], reverse=True)[:max_size]
    )

    return sorted_ranking


# TRANSCRIPT SELECTOR
def transcript_selector(sorted_ranking):
    """
    Select transcript files corresponding to the top-ranked chunks.

    Args:
        sorted_ranking (dict): Dictionary of chunk scores sorted in descending order

    Returns:
        list: List of transcript contents for the top-ranked chunks
    """

    # Selected transcripts list
    selected_transcripts = []

    # Iterate through the sorted ranking
    for chunk_filename in sorted_ranking.keys():
        # Extract the number from the chunk score filename
        match = re.search(r"chunk_score_(\d+)\.json", chunk_filename)
        if match:
            chunk_number = match.group(1)

            # Construct the corresponding transcript filename
            transcript_filename = f"sliding_transcript_{chunk_number}.json"
            transcript_path = os.path.join(sliding_transcript_dir, transcript_filename)

            selected_transcripts.append(transcript_filename)
            print(transcript_path)

    return selected_transcripts


# VIDEO SEGMENT EXTRACTOR
def extract_video_segments(selected_transcripts):
    """
    Extract start and end times for each transcript chunk.

    Args:
        transcript_dir (str): Directory containing transcript JSON files

    Returns:
        dict: A dictionary mapping transcript filenames to their start and end times
    """
    # Initialize video segments dictionary
    video_segments = {}

    # Iterate through all selected transcript 
    for filename in selected_transcripts:
        if filename.startswith("sliding_transcript_") and filename.endswith(".json"):
            # Construct full file path
            file_path = os.path.join(sliding_transcript_dir, filename)

            try:
                # Read the transcript file
                with open(file_path, "r") as file:
                    transcript = json.load(file)

                # Check if 'words' list exists and is not empty
                if transcript.get("words") and len(transcript["words"]) > 0:
                    # Get start time of the first word
                    start_time = transcript["words"][0]["start"]

                    # Get end time of the last word
                    end_time = transcript["words"][-1]["end"]

                    # Store in video_segments dictionary
                    video_segments[filename] = {"start": start_time, "end": end_time}

            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error processing {filename}: {e}")

    return video_segments


def curation():
    # step 1
    ranking = rank_chunk_scores(max_size=5)
    print("Top 2 Chunk Score Ranking:")
    for filename, score in ranking.items():
        print(f"{filename}: {score}")

    # step 2
    selected_transcripts = transcript_selector(ranking)
    print(f"Selected Transcripts :",selected_transcripts)

    # step 3
    segments = extract_video_segments(selected_transcripts)
    output_path = os.path.join(script_dir, "assets/temp/curated_video_segments.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as json_file:
        json.dump(segments, json_file, indent=4)
    
if __name__ == "__main__":
    curation()
