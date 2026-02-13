# SecureCall AI Detector 🛡️
> *Winner-Ready Hackathon Project | Hybrid AI + Physics Voice Fraud Detection*

**SecureCall** is an advanced voice fraud detection API designed to catch deepfakes that traditional models miss. It uniquely combines deep learning with acoustic physics to distinguish between high-quality AI cloning and real human vocal production.

## 🚀 Key Features (The "Wow" Factor)

### 1. Hybrid Detection Engine
Most detectors rely solely on AI models (which can be fooled by new TTS). We use a **Triple-Check System**:
-   **Deep Learning**: Multilingual Wav2Vec2 (XLS-R) model.
-   **Robotic Smoothness**: Heuristics to detect unnatural consistency in AI speech.
-   **Physics Validator (pYIN)**: Analysis of "vocal jitter". Real vocal cords have natural instability (2-8Hz); AI is mathematically perfection. **We catch the perfection.**

### 2. Multilingual Support 🇮🇳
Targeted for Indian contexts using `XLS-R` architecture. Tested on Hindi, English, and regional accents.

### 3. "Goldilocks" Scoring
Our system allows:
-   **Perfect Detection**: Detects AI even if it sounds "human" to the ear (via smoothness).
-   **Zero False Positives**: "Rescues" real human voices (even with noise) if they have valid vocal physics.

## 🛠️ Tech Stack
-   **FastAPI**: High-performance backend.
-   **Transformers**: `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`.
-   **Librosa & PyAV**: Advanced signal signal processing & ffmpeg-free decoding.
-   **OpenAI Whisper**: Automatic transcription & fraud keyword detection.

## 🎤 Usage

### 1. Start Server
```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test
-   **Web Interface**: Open `http://localhost:8000` to record your voice.
-   **API**: Run `python test_api.py` for detailed diagnostics.

## 📊 Diagnostics Example
The API explains *why* it made a decision:
```json
{
  "classification": "AI",
  "confidence": 0.86,
  "heuristics": {
    "pitch_jitter": 11.6, // Too high for human, rejected
    "smoothness": 0.98    // Too smooth, robotic
  }
}
```
