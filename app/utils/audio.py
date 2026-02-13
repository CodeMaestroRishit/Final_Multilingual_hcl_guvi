import base64
import io
import librosa
import numpy as np
import requests
import soundfile as sf
import torch
import av
from fastapi import HTTPException

def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000, max_duration: float = None):
    """
    Decodes audio bytes (MP3, WAV, etc.) and resamples to target_sr.
    Returns (numpy_array, metadata_dict).
    """
    metadata = {}
    try:
        # Try PyAV first for metadata (fast header read)
        try:
            with av.open(io.BytesIO(audio_bytes)) as container:
                metadata = dict(container.metadata)
        except Exception:
            pass # Continue to loading if metadata fails

        # Fallback to PyAV for decoding if needed, but try Librosa/Soundfile first for speed
        # using duration param to only load what we need (HUGE SPEEDUP for long files)
        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=target_sr,
            mono=True,
            duration=max_duration,
        )
        if y.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio is empty")
        return y, metadata
    except Exception as e:
        # Fallback to PyAV (ffmpeg-less decoding)
        print(f"Librosa load failed: {e}. Trying PyAV...")
        try:
            y, metadata = decode_with_pyav(audio_bytes, target_sr, max_duration)
            if y.size == 0:
                raise HTTPException(status_code=400, detail="Decoded audio is empty")
            return y, metadata
        except Exception as av_e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio (Librosa & PyAV): {str(e)} | {str(av_e)}")

def decode_with_pyav(audio_bytes: bytes, target_sr: int = 16000, max_duration: float = None):
    """
    Decodes audio using PyAV. Returns (y, metadata).
    """
    try:
        container = av.open(io.BytesIO(audio_bytes))
        metadata = dict(container.metadata)
        stream = container.streams.audio[0]
        
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
        
        # Calculate max samples
        max_samples = int(target_sr * max_duration) if max_duration else None
        
        audio_data = []
        total_samples = 0
        
        for frame in container.decode(stream):
            # Resample frame
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                # To numpy
                chunk = resampled_frames.to_ndarray() # shape (1, samples)
                chunk_len = chunk.shape[1]
                
                # Check limit
                if max_samples and (total_samples + chunk_len > max_samples):
                    # Slice the last chunk
                    remaining = max_samples - total_samples
                    audio_data.append(chunk[0][:remaining])
                    total_samples += remaining
                    break
                else:
                    audio_data.append(chunk[0])
                    total_samples += chunk_len
                
        if not audio_data:
            raise ValueError("No audio data decoded")
            
        y = np.concatenate(audio_data)
        return y, metadata
    except Exception as e:
        raise ValueError(f"PyAV decoding failed: {e}")

def decode_base64_audio(b64_string: str) -> bytes:
    try:
        # Handle data URI scheme if present (e.g. "data:audio/mp3;base64,...")
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        return base64.b64decode(b64_string)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 string: {str(e)}")

def download_audio_from_url(url: str) -> bytes:
    try:
        # Wikimedia and others require a User-Agent
        headers = {"User-Agent": "VoiceDetectorAPI/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")

def process_audio_input(audio_base64: str | None, audio_url: str | None, max_duration: float = 6.0):
    """
    Orchestrates loading audio from either base64 or URL.
    Returns (raw_waveform, metadata).
    """
    if audio_base64:
        audio_bytes = decode_base64_audio(audio_base64)
    elif audio_url:
        audio_bytes = download_audio_from_url(audio_url)
    else:
        raise HTTPException(status_code=400, detail="No audio provided")
    
    return load_audio_from_bytes(audio_bytes, max_duration=max_duration)
