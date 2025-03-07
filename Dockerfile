FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (ffmpeg and imagemagick)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Copy a custom ImageMagick policy file to allow necessary operations
COPY policy.xml /etc/ImageMagick-6/policy.xml

# Copy requirements file first (optimization for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["fastapi","run", "main.py", "--host", "0.0.0.0", "--port", "8000"]