# Deployment Guide for Railway

This guide explains how to deploy the AI Voice Detector API to [Railway](https://railway.app/).

## Prerequisites

- A Railway account.
- This repository pushed to GitHub.

## Deployment Steps

1.  **Create a New Project**: On Railway, click "New Project" and select "Deploy from GitHub repo".
2.  **Select Repository**: Choose this repository.
3.  **Configure Environment Variables**:
    Go to the **Variables** tab in your Railway service and add the following:
    - `SARVAM_API_KEY`: (Required) Your API key from Sarvam AI.
    - `API_KEY`: (Optional) Your custom API key for the `X-API-Key` header. Defaults to `demo_key_123` if not provided.
4.  **Project Settings**:
    - Railway will automatically detect the `Dockerfile` and build the image.
    - The `PORT` variable is automatically injected by Railway.
5.  **Deploy**: Railway will trigger the build and deployment automatically.

## API Documentation

Once deployed, the following endpoints will be available:

- `GET /health`: Health check endpoint.
- `POST /detect`: Main detection endpoint (requires `X-API-Key` header).
- `POST /api/voice-detection`: Strict hackathon format endpoint (requires `X-API-Key` header).

## Testing the Deployment

You can use the `test_api.py` script by updating the `API_URL` to your Railway service URL:

```python
API_URL = "https://your-service-name.up.railway.app"
API_KEY = "your_configurated_api_key"
```
