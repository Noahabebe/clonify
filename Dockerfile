# Use an official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for FFmpeg and OpenCV compatibility
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose port 5000
EXPOSE 5000

# Command to run Gunicorn with Gevent workers
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
