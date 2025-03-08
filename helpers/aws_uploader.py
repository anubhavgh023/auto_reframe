import os
import boto3 
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Function to upload to S3 and generate presigned URL
def upload_to_s3(file_path: str, duration: int) -> str:
    # Generate a timestamped filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_time}-{duration}.mp4"

    s3_path = f"uploads/{file_name}"
    bucket_name = os.getenv("S3_BUCKET_NAME")
    try:
        # Upload the file to S3
        s3_client.upload_file(file_path, bucket_name, s3_path,ExtraArgs={"ContentType":"video/mp4"})

        # Generate a presigned URL valid for 1 hour (3600 seconds)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_path},
            ExpiresIn=3600,  # 1 hour
        )
        return presigned_url
    except Exception as e:
        return f"ERROR: {e}"
