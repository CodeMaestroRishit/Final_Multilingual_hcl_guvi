# Use lightweight Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg and libsndfile1 for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model
# This layer will be cached unless requirements.txt changes
RUN python -c "from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification; \
    model_name = 'Gustking/wav2vec2-large-xlsr-deepfake-audio-classification'; \
    print(f'Pre-downloading model: {model_name}...'); \
    Wav2Vec2FeatureExtractor.from_pretrained(model_name); \
    AutoModelForAudioClassification.from_pretrained(model_name); \
    print('Model downloaded successfully.')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
