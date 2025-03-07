# import os
# import json
# from dotenv import load_dotenv
# from openai import OpenAI
# from tqdm import tqdm

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")

# # Paths
# SLIDING_TRANSCRIPTS_DIR = "./assets/audio/sliding_transcript_chunks/"
# SCORES_DIR = "./assets/scores/"

# # Ensure scores directory exists
# os.makedirs(SCORES_DIR, exist_ok=True)


# def generate_scoring_prompt(transcript_text):
#     """Generate a formatted scoring prompt with the transcript text included."""
#     return f"""
# You are an expert content analyst specializing in short-form viral videos.
# Your task is to evaluate a transcript segment for potential "viral" appeal.

# Analyze the following text:
# "{transcript_text}"

# We want you to check for these 8 categories that contribute to virality:

# 1. Hook Signals (max 15 points):
#    - Looks for strong opening lines or phrases that quickly grab attention.

# 2. Conflict or Problem (max 15 points):
#    - Looks for statements of tension, challenge, or obstacles.

# 3. Emotional/Exciting Moments (max 15 points):
#    - Looks for evidence of strong positive or negative emotion, excitement, or shock.

# 4. Twist or Resolution (max 15 points):
#    - Indicates a surprising turn of events or a clear, satisfying payoff.

# 5. Foreshadow / Teaser (max 10 points):
#    - Mentions of a future reveal or reason to "stick around."

# 6. Step-by-Step / List Structure (max 10 points):
#    - Presence of enumerated steps or instructions.

# 7. Personal / Relatable Anecdote (max 10 points):
#    - The speaker references personal stories or experiences that create emotional resonance.

# 8. Shareability / "Wow" Factor (max 10 points):
#    - Overall uniqueness, entertainment value, or "wow" that would make someone want to share it.

# Return ONLY a valid JSON structure like this:

# {{
#   "HookSignalsScore": 0,
#   "ConflictScore": 0,
#   "EmotionScore": 0,
#   "TwistScore": 0,
#   "ForeshadowScore": 0,
#   "StepsScore": 0,
#   "PersonalAnecdoteScore": 0,
#   "ShareabilityScore": 0,
#   "TotalScore": 0,
#   "Justification": {{
#     "HookSignalsNotes": "EXPLANATION",
#     "ConflictNotes": "EXPLANATION",
#     "EmotionNotes": "EXPLANATION",
#     "TwistNotes": "EXPLANATION",
#     "ForeshadowNotes": "EXPLANATION",
#     "StepsNotes": "EXPLANATION",
#     "PersonalAnecdoteNotes": "EXPLANATION",
#     "ShareabilityNotes": "EXPLANATION"
#   }}
# }}
# """


# def process_transcript(transcript_path):
#     """Process a single transcript file and generate a viral score."""
#     try:
#         # Read the transcript
#         with open(transcript_path, "r", encoding="utf-8") as f:
#             transcript_data = json.load(f)

#         # Extract text
#         transcript_text = transcript_data.get("text", "").strip()

#         # Skip if no text
#         if not transcript_text:
#             print(f"No text in {transcript_path}")
#             return None

#         # Initialize OpenAI client
#         client = OpenAI(api_key=API_KEY)

#         # Generate scoring prompt dynamically
#         scoring_prompt = generate_scoring_prompt(transcript_text)

#         # Send to OpenAI
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a viral content analysis expert.",
#                 },
#                 {"role": "user", "content": scoring_prompt},
#             ],
#             response_format={"type": "json_object"},
#         )

#         # Parse the response
#         score_data = json.loads(response.choices[0].message.content)

#         return score_data

#     except Exception as e:
#         print(f"Error processing {transcript_path}: {e}")
#         return None


# def process_transcript_main():
#     # Get all transcript files
#     transcript_files = sorted(
#         [
#             f
#             for f in os.listdir(SLIDING_TRANSCRIPTS_DIR)
#             if f.startswith("sliding_transcript_") and f.endswith(".json")
#         ]
#     )

#     # Process each transcript
#     for transcript_file in tqdm(transcript_files, desc="Processing Sliding Transcripts"):
#         # Full paths
#         transcript_path = os.path.join(SLIDING_TRANSCRIPTS_DIR, transcript_file)

#         # Extract chunk number
#         chunk_num = transcript_file.split("_")[-1].split(".")[0]
#         score_file_path = os.path.join(SCORES_DIR, f"chunk_score_{chunk_num}.json")

#         # Process transcript
#         score_data = process_transcript(transcript_path)

#         # Save score if successful
#         if score_data:
#             with open(score_file_path, "w", encoding="utf-8") as f:
#                 json.dump(score_data, f, indent=4)

#     print("Scoring complete!")


# if __name__ == "__main__":
#     process_transcript_main()

# --------------------------------------
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
print("----------------------------------")
print("----------------------------------")
print(API_KEY)
print("----------------------------------")
print("----------------------------------")

# Paths
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
SLIDING_TRANSCRIPTS_DIR = os.path.join(
    script_dir, "assets/audio/sliding_transcript_chunks/"
)
SCORES_DIR = os.path.join(script_dir, "assets/scores/")

# Ensure scores directory exists
os.makedirs(SCORES_DIR, exist_ok=True)

# Batch size for processing
BATCH_SIZE = 5


def generate_scoring_prompt(transcript_text):
    """Generate a formatted scoring prompt with the transcript text included."""
    return f"""
You are an expert content analyst specializing in short-form viral videos. 
Your task is to evaluate a transcript segment for potential "viral" appeal.

Analyze the following text:
"{transcript_text}"

We want you to check for these 8 categories that contribute to virality:

1. Hook Signals (max 15 points):
   - Looks for strong opening lines or phrases that quickly grab attention.

2. Conflict or Problem (max 15 points):
   - Looks for statements of tension, challenge, or obstacles.

3. Emotional/Exciting Moments (max 15 points):
   - Looks for evidence of strong positive or negative emotion, excitement, or shock.

4. Twist or Resolution (max 15 points):
   - Indicates a surprising turn of events or a clear, satisfying payoff.

5. Foreshadow / Teaser (max 10 points):
   - Mentions of a future reveal or reason to "stick around."

6. Step-by-Step / List Structure (max 10 points):
   - Presence of enumerated steps or instructions.

7. Personal / Relatable Anecdote (max 10 points):
   - The speaker references personal stories or experiences that create emotional resonance.

8. Shareability / "Wow" Factor (max 10 points):
   - Overall uniqueness, entertainment value, or "wow" that would make someone want to share it.

Return ONLY a valid JSON structure like this:

{{
  "HookSignalsScore": 0,
  "ConflictScore": 0,
  "EmotionScore": 0,
  "TwistScore": 0,
  "ForeshadowScore": 0,
  "StepsScore": 0,
  "PersonalAnecdoteScore": 0,
  "ShareabilityScore": 0,
  "TotalScore": 0,
  "Justification": {{
    "HookSignalsNotes": "EXPLANATION",
    "ConflictNotes": "EXPLANATION",
    "EmotionNotes": "EXPLANATION",
    "TwistNotes": "EXPLANATION",
    "ForeshadowNotes": "EXPLANATION",
    "StepsNotes": "EXPLANATION",
    "PersonalAnecdoteNotes": "EXPLANATION",
    "ShareabilityNotes": "EXPLANATION"
  }}
}}
"""


def process_transcript(
    transcript_file, transcript_dir=SLIDING_TRANSCRIPTS_DIR, scores_dir=SCORES_DIR
):
    """Process a single transcript file and generate a viral score."""
    try:
        # Full path
        transcript_path = os.path.join(transcript_dir, transcript_file)

        # Extract chunk number
        chunk_num = transcript_file.split("_")[-1].split(".")[0]
        score_file_path = os.path.join(scores_dir, f"chunk_score_{chunk_num}.json")

        # Skip if already processed
        if os.path.exists(score_file_path):
            return f"Skipped {transcript_file} (already processed)"

        # Read the transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Extract text
        transcript_text = transcript_data.get("text", "").strip()

        # Skip if no text
        if not transcript_text:
            return f"No text in {transcript_path}"

        # Initialize OpenAI client
        client = OpenAI(api_key=API_KEY)

        # Generate scoring prompt dynamically
        scoring_prompt = generate_scoring_prompt(transcript_text)

        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a viral content analysis expert.",
                },
                {"role": "user", "content": scoring_prompt},
            ],
            response_format={"type": "json_object"},
        )

        # Parse the response
        score_data = json.loads(response.choices[0].message.content)

        # Save score
        with open(score_file_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, indent=4)

        return f"Processed {transcript_file}"

    except Exception as e:
        return f"Error processing {transcript_file}: {e}"


def batch_process_files(file_batch):
    """Process a batch of files and return results."""
    results = []
    for file in file_batch:
        result = process_transcript(file)
        results.append(result)
    return results


def process_transcript_main():
    # Get all transcript files
    transcript_files = sorted(
        [
            f
            for f in os.listdir(SLIDING_TRANSCRIPTS_DIR)
            if f.startswith("sliding_transcript_") and f.endswith(".json")
        ]
    )

    # Split files into batches
    batches = [
        transcript_files[i : i + BATCH_SIZE]
        for i in range(0, len(transcript_files), BATCH_SIZE)
    ]

    # Set up multiprocessing
    pool = mp.Pool(processes=min(mp.cpu_count(), BATCH_SIZE))

    # Process batches in parallel with progress bar
    results = []
    for batch_results in tqdm(
        pool.imap(batch_process_files, batches),
        total=len(batches),
        desc="Processing Batches",
    ):
        results.extend(batch_results)

    # Close the pool
    pool.close()
    pool.join()

    # Print results summary
    success_count = sum(
        1 for result in results if "Error" not in result and "Skipped" not in result
    )
    skip_count = sum(1 for result in results if "Skipped" in result)
    error_count = sum(1 for result in results if "Error" in result)

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Skipped (already processed): {skip_count} files")
    print(f"Errors: {error_count} files")

    # Print errors if any
    if error_count > 0:
        print("\nErrors encountered:")
        for result in results:
            if "Error" in result:
                print(f"  - {result}")


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method("spawn", force=True)
    process_transcript_main()
