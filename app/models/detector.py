import os
import numpy as np
import librosa
import numpy as np
import librosa
# import noisereduce as nr (Disabled: causes artifacts)
import io
import requests
from fastapi import HTTPException

def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")

ML_DISABLED = _env_truthy("DISABLE_ML")

if not ML_DISABLED:
    import torch
    import torch.nn.functional as F
    from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, pipeline
else:
    # Avoid importing torch/transformers entirely in keyword-only mode.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Wav2Vec2FeatureExtractor = None  # type: ignore[assignment]
    AutoModelForAudioClassification = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]

class VoiceDetector:
    _instance = None
    
    def __init__(self):
        if ML_DISABLED or torch is None:
            raise RuntimeError("ML is disabled (set DISABLE_ML=0 to enable model inference)")

        from torch.nn import CosineSimilarity
        self.cos_sim = CosineSimilarity(dim=1, eps=1e-6)

        print("Initializing Detection Pipeline...")
        
        # 1. Primary AI vs Human detection (language-agnostic)
        # Using a multilingual XLS-R based model for better Hindi/non-English support
        self.detector_model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        print(f"Loading AI Detector: {self.detector_model_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.detector_model_name
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.detector_model_name
        )
        self.model.eval()
        
        # 2. Transcription and Translation (DISABLED FOR SPEED)
        self.whisper_model_name = None 
        self.transcriber = None
        
        # ── Refined Fraud Categories ──────────────────────────────────
        # HIGH SEVERITY CATEGORIES
        self.risk_categories = {
            "secrets": [
                "otp", "one time password", "tell me otp", "share otp", "send otp",
                "cvv", "pin", "atm pin", "upi pin",
                "password", "netbanking", "credentials",
                "screen share", "anydesk", "teamviewer",
                "sms code", "verification code", "tell the code",
            ],
            "threats": [
                "account will be blocked", "account blocked", "sim will be deactivated",
                "final notice", "last chance", "immediately", "urgent",
                "before lines close", "within 1 hour", "within 2 hours",
                "within 24 hours", "within 23 hours",
            ],
            "prizes": [
                "you have won", "you won", "congratulations you are selected",
                "congratulations", "claim your prize", "prize", "cash reward",
                "reward", "lottery", "lucky draw", "bonus caller",
                "free holiday", "voucher",
            ],
            "payments": [
                "pay now", "pay on this", "send money", "transfer money",
                "processing fee", "logistics charge", "deposit to receive",
            ],
            "premium": [
                "premium", "landline", "charged per", "10ppm", "150p/min", "090", "087"
            ],
            # MEDIUM SEVERITY / CONTEXT CATEGORIES
            "institutions": [
                "bank", "sbi", "hdfc", "icici", "axis", "union bank", 
                "customer care", "manager", "verification", "statement", "kyc", 
                "bhim", "upi upgrade", "account"
            ],
            "cta": [
                "call now", "call this number", "call immediately",
                "click link", "click here", "visit website",
            ],
            "generic": [
                "suspicious", "security", "verify", "identity", "activity"
            ]
        }

        # Multilingual mapping for core HIGH risk terms
        self.multilingual_high = {
            "hindi_devanagari": ["ओटीपी", "पिन", "पासवर्ड", "ब्लॉक", "तुरंत", "लॉटरी"],
            "hindi_roman": ["otp", "pin", "band", "jald", "jaldi"],
            "tamil_native": ["ஓடீபி", "பின்", "பாஸ்வர்ட்", "அவசரம்", "பரிசு", "ஒடிபி"],
            "telugu_native": ["ఓటీపీ", "పిన్", "పాస్వర్డ్", "అత్యవసరం", "బహుమతి"],
            "malayalam_native": ["ഒടിപി", "പിൻ", "പാസ്‌വേർഡ്", "അടിയന്തിരം", "സമ്மானம்"],
        }
        
        # Flattened list for the existing _check_keywords logic (backward compatibility)
        self.high_risk_keywords = {
            "english": (
                self.risk_categories["secrets"] + 
                self.risk_categories["threats"] + 
                self.risk_categories["prizes"] + 
                self.risk_categories["payments"] + 
                self.risk_categories["premium"]
            ),
            **self.multilingual_high
        }
        
        self.low_risk_keywords = {
            "english": (
                self.risk_categories["institutions"] + 
                self.risk_categories["cta"] + 
                self.risk_categories["generic"]
            ),
            "hindi_devanagari": ["बैंक", "खाता", "वेरिफाई", "नंबर", "कार्ड"],
            "hindi_roman": ["bank", "khata", "verify", "number"],
            "tamil_native": ["கணக்கு", "வங்கி", "பேங்க்", "அக்கவுண்ட்", "வெரிஃபை", "கஸ்டமர்", "கஸ்டமர் கேர்", "அன்ஆத்தரைஸ்டா"],
            "telugu_native": ["ఖాతా", "బ్యాంக்"],
            "malayalam_native": ["അക്കൗണ്ട്", "ബാങ്ക്"],
        }
        
        print("AI Detector loaded successfully. (Whisper/Fraud disabled for performance)")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_audio(self, input_audio):
        """
        Download or decode the audio from URL or Base64/Bytes.
        Returns floating point audio array.
        """
        # If input is URL
        if isinstance(input_audio, str) and input_audio.startswith("http"):
            response = requests.get(input_audio)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        # If input is bytes-like (file or base64 decoded bytes)
        elif isinstance(input_audio, (bytes, bytearray, io.BytesIO)):
             if isinstance(input_audio, (bytes, bytearray)):
                 audio_bytes = io.BytesIO(input_audio)
             else:
                 audio_bytes = input_audio
        elif isinstance(input_audio, np.ndarray):
             # Already loaded audio
             return input_audio, 16000 # Assume 16k if passed from utils, or check logic
        else:
            # Assume it's a file path or direct numpy (if passed locally)
            audio_bytes = input_audio

        # Load with Librosa
        try:
             # librosa.load can handle path or file-like object
             y, sr = librosa.load(audio_bytes, sr=None)
             return y, sr
        except Exception as e:
             raise ValueError(f"Failed to load audio: {e}")

    def _preprocess_audio(self, y, sr):
        """
        Convert to mono, 16 kHz.
        Apply: noise reduction, silence trimming, normalization to -1..1.
        Return processed audio and new sample rate (16000).
        """
        target_sr = 16000
        
        # 1. Convert to mono and resample to 16kHz
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Ensure mono (librosa.load defaults to mono=True, but just in case)
        if y.ndim > 1:
            y = librosa.to_mono(y)
            
        # 2. Noise Reduction
        # Using stationary noise reduction
        # noisy reducation causing artifacts on clean audio? 
        # y = nr.reduce_noise(y=y, sr=target_sr)
        
        # 3. Silence Trimming
        # top_db=20 is a common default, adjusting as needed. Prompt didn't specify db.
        y, _ = librosa.effects.trim(y)
        
        # 4. Normalization to -1..1
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
            
        return y, target_sr

    def _chunk_audio(self, y, sr, chunk_duration=30):
        """
        If audio is longer than 30 seconds, split into chunks.
        """
        duration = len(y) / sr
        chunks = []
        if duration > chunk_duration:
            samples_per_chunk = int(chunk_duration * sr)
            total_samples = len(y)
            for start in range(0, total_samples, samples_per_chunk):
                end = min(start + samples_per_chunk, total_samples)
                chunks.append(y[start:end])
        else:
            chunks.append(y)
        return chunks

    def _calculate_smoothness(self, embeddings) -> float:
        """
        Calculates temporal smoothness.
        AI voices tend to have higher frame-to-frame cosine similarity (less 'jitter').
        """
        if embeddings.shape[1] < 2:
            return 0.0
            
        # Compare all frames with their next frame
        similarity = self.cos_sim(embeddings[0, :-1, :], embeddings[0, 1:, :])
        return float(similarity.mean().item())

    def _calculate_snr(self, y: np.ndarray) -> float:
        """
        Calculates Signal-to-Noise Ratio (SNR) of the audio.
        High SNR (> 60dB) -> Studio quality (likely AI or studio rec).
        Lower SNR (< 30dB) -> Natural background noise (likely Human).
        """
        # Simple energy-based estimation
        # Assume lowest 10% energy frames are "noise" floor
        if len(y) < 100:
            return 0.0
            
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max) # 0 dB is max
        
        # Sort frame energies
        sorted_db = np.sort(rms_db[0])
        
        # Estimate noise floor (average of lowest 10% of frames)
        # Avoid silence trimming artifacts by taking 5th to 15th percentile
        noise_idx = int(len(sorted_db) * 0.1)
        if noise_idx == 0: noise_idx = 1
        noise_floor_db = np.mean(sorted_db[:noise_idx])
        
        # Signal power (average of top 20% of frames)
        signal_idx = int(len(sorted_db) * 0.8)
        signal_power_db = np.mean(sorted_db[signal_idx:])
        
        snr_value = signal_power_db - noise_floor_db
        return float(snr_value)

    def _calculate_pitch_score(self, y, sr):
        """
        Estimates 'Human-ness' based on Pitch (F0) variance and jitter.
        Real voices have higher pitch standard deviation and frame-to-frame jitter.
        Returns score 0.0 (Robotic) to 1.0 (Very Human).
        """
        try:
            # Estimate pitch using pyin (Probabilistic YIN) - Robust to noise
            # fmin=50Hz (Deep male), fmax=1000Hz (High female/Screams)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            
            # Filter unvoiced
            f0 = f0[~np.isnan(f0)]
            
            if len(f0) < 10:
                print("DEBUG: Pitch Analysis -> Too few voiced frames.")
                return 0.0, 0.0, 0.0
                
            # 1. Pitch Standard Deviation (Intonation richness)
            pitch_std = np.std(f0)
            
            # 2. Jitter Proxy (Frame-to-frame absolute difference)
            jitter = np.mean(np.abs(np.diff(f0)))
            
            # Normalize (Heuristics) - "Goldilocks Trapezoid" based on pYIN
            # Human Jitter: 2Hz - 8Hz is the "Sweet Spot".
            # < 1Hz: Robotic.
            # > 10Hz: Unnatural/Noisy.
            
            if jitter < 1.0:
                score_jitter = 0.0
            elif 1.0 <= jitter < 2.0:
                score_jitter = (jitter - 1.0) # Ramp 0->1
            elif 2.0 <= jitter <= 8.0:
                score_jitter = 1.0 # Sweet spot
            elif 8.0 < jitter < 12.0:
                score_jitter = 1.0 - ((jitter - 8.0) / 4.0) # Ramp 1->0
            else: # > 12.0
                score_jitter = 0.0
                 
            # Std Score
            if pitch_std < 5.0:
                score_std = 0.0 # Monotone
            else:
                score_std = min(1.0, pitch_std / 20.0) # 25Hz std is good
            
            # Weight Jitter higher (80%) because intonation (std) is easy to fake
            final_score = (score_std * 0.2) + (score_jitter * 0.8)
            
            print(f"DEBUG: Pitch Analysis -> Std={pitch_std:.2f} (Score={score_std:.2f}), Jitter={jitter:.2f} (Score={score_jitter:.2f}) -> Final={final_score:.2f}")
            
            return final_score, pitch_std, jitter
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0

    def _check_keywords(self, transcript: str):
        """
        Checks for fraud keywords in the transcript.
        Returns (list of found keywords, detected language, high_count, low_count).
        Keywords are tagged with their risk level: [HIGH] or [LOW].
        Uses whole-word matching to avoid false positives (e.g., 'pin' in 'capping').
        """
        import re
        
        if not transcript:
            return [], "unknown", 0, 0
            
        transcript_lower = transcript.lower()
        found = []
        high_count = 0
        low_count = 0
        seen = set()
        
        def _use_word_boundaries(kw: str) -> bool:
            # Python's \\b word boundary is unreliable for many non-Latin scripts.
            # Use it only for ASCII keywords; otherwise fall back to substring matching.
            return kw.isascii()

        # Check HIGH risk keywords
        for lang, keywords in self.high_risk_keywords.items():
            for kw in keywords:
                if _use_word_boundaries(kw):
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    hit = re.search(pattern, transcript_lower) is not None
                else:
                    hit = kw in transcript_lower
                if hit:
                    tag = f"{kw} ({lang}) [HIGH]"
                    if tag not in seen:
                        found.append(tag)
                        seen.add(tag)
                        high_count += 1
        
        # Check LOW risk keywords
        for lang, keywords in self.low_risk_keywords.items():
            for kw in keywords:
                if _use_word_boundaries(kw):
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    hit = re.search(pattern, transcript_lower) is not None
                else:
                    hit = kw in transcript_lower
                if hit:
                    tag = f"{kw} ({lang}) [LOW]"
                    if tag not in seen:
                        found.append(tag)
                        seen.add(tag)
                        low_count += 1
                    
        return found, "multilingual", high_count, low_count

    def detect_fraud(self, input_audio, metadata=None, transcript=None):
        # Initialize diagnostics
        smoothness = 0.0
        time_variance = 0.0
        heuristic_score = 0.0
        probs = None
        pitch_score = 0.0
        snr_score = 0.0
        metadata_hit = False
        metadata_explanation = ""
        metadata_note = None
        
        # Initialize text fields
        detected_language = "unknown"
        transcription = ""
        english_translation = ""
        overall_risk = "LOW"
        found_keywords = []
        
        # --- Metadata Short-Circuit (Instant Speed + High Accuracy) ---
        if metadata:
            encoder = metadata.get("encoder", "").lower()
            handler = metadata.get("handler_name", "").lower()
            
            # "Lavf" = Libavformat (FFmpeg). Almost all API-generated audio uses this.
            # "LAME" = Encoder often used in programmatic generation.
            # Real recordings usually have "iTunes", "Android", or no encoder tag.
            # Real recordings usually have "iTunes", "Android", or no encoder tag.
            if "lavf" in encoder or "lavc" in encoder or "google" in encoder:
                print(f"DEBUG: METADATA HIT! Encoder={encoder}. Marking as AI but continuing analysis.")
                metadata_hit = True
                metadata_explanation = f"Metadata analysis detected programmatic encoder: {metadata.get('encoder')}"

        # --- Audio Loading & Preprocessing ---
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)
        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio contained no samples after preprocessing")
        
        # --- Primary AI vs Human detection ---
        # TURBO MODE: 2 seconds max for Railway hackathon timeout.
        # 16000 Hz * 2 seconds = 32000 samples
        max_samples = 16000 * 2
        if len(y) > max_samples:
            y = y[:max_samples]
            
        # Re-chunking is trivial now (it will be 1 chunk)
        chunks = [c for c in self._chunk_audio(y, sr) if len(c) > 0]
        if not chunks:
            raise HTTPException(status_code=400, detail="Audio contained no decodable frames")
        
        ai_probs = []
        
        for chunk in chunks:
            # Prepare inputs
            # Wav2Vec2 inputs
            # Processor requires list of numpy arrays, but we usually pass one by one or batched.
            # padding=True/False depends on if we batch. Here iterative.
            inputs = self.feature_extractor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            print(f"DEBUG: Probs: {probs[0].tolist()}, Labels: {self.model.config.id2label}")
            
            # Check model labels
            # Confirmed via debug: {0: 'real', 1: 'fake'}
            # Index 1 is AI/Fake.
            p_ai_chunk = probs[0][1].item()
            
            # Robustness check: if model has id2label, we could use it.
            ai_probs.append(p_ai_chunk)
            
        # Aggregate
        p_ai_model = sum(ai_probs) / len(ai_probs) if ai_probs else 0.0
        
        # --- TURBO+ MODE: Model-first, with smart fallbacks ---
        # Layer 1: Model (always runs, ~200ms)
        # Layer 2: Metadata (instant, hard signal)
        # Layer 3: Pitch (only when model is uncertain, ~50ms)
        heuristic_score = 0.0
        smoothness = 0.0
        time_variance = 0.0
        pitch_score = 0.0
        p_std = 0.0
        p_jitter = 0.0
        snr_val = 0.0
        snr_score = 0.0
        
        print(f"DEBUG: TURBO+ MODE -> ModelProb={p_ai_model:.3f}")
        
        # Base: Trust the model
        final_p_ai = p_ai_model
        
        # --- Layer 2: Metadata (instant, can't be spoofed by noise) ---
        if metadata_hit:
            metadata_note = metadata_explanation or "Suspicious encoder metadata detected"
            # Stronger boost: 20% instead of 10%
            final_p_ai = min(1.0, final_p_ai + 0.20)
            print(f"DEBUG: Metadata boost -> {p_ai_model:.3f} -> {final_p_ai:.3f}")
        
        # --- Layer 3: Conditional Pitch (only when model is uncertain) ---
        model_uncertain = (0.35 < p_ai_model < 0.65)
        if model_uncertain:
            print(f"DEBUG: Model uncertain ({p_ai_model:.3f}), running pitch analysis...")
            pitch_score, p_std, p_jitter = self._calculate_pitch_score(y, sr)
            
            # If pitch says robotic (low score) -> push toward AI
            if pitch_score < 0.3:
                final_p_ai = min(1.0, final_p_ai + 0.15)
                print(f"DEBUG: Pitch says robotic ({pitch_score:.2f}) -> boosted to {final_p_ai:.3f}")
            # If pitch says human (high score) AND model leans human -> reinforce
            elif pitch_score > 0.7 and p_ai_model < 0.5:
                final_p_ai = max(0.05, final_p_ai - 0.15)
                print(f"DEBUG: Pitch says human ({pitch_score:.2f}) -> reduced to {final_p_ai:.3f}")
        
        classification = "AI" if final_p_ai > 0.5 else "Human"
        confidence = max(final_p_ai, 1 - final_p_ai)
        p_ai = final_p_ai
        
        # --- Explanation String ---
        parts = []
        parts.append(f"AI probability {round(p_ai, 2)}")
        parts.append(f"Deepfake detector classified as {classification}")
        if metadata_note:
             parts.append(metadata_note)
        if model_uncertain and pitch_score > 0:
             if pitch_score > 0.7:
                  parts.append("Natural human pitch variations detected")
             elif pitch_score < 0.3:
                  parts.append("Robotic pitch patterns detected")
        
        # --- Transcript & Keyword Analysis ---
        found_keywords = []
        if transcript:
            transcription = transcript
            found_keywords, _, high_c, low_c = self._check_keywords(transcription)
            print(f"🔍 KEYWORDS: Checked transcript ({len(transcription)} chars) -> Found: {found_keywords}")
            
            if found_keywords:
                if high_c > 0:
                    overall_risk = "HIGH"
                else:
                    overall_risk = "MEDIUM"
                parts.append(f"Fraud keywords detected: {', '.join(found_keywords)}")
        
        explanation = ", ".join(parts)
        
        # Calculate audio duration for diagnostics
        audio_duration_seconds = round(len(y) / sr, 2)
        
        return {
            "classification": classification,
            "confidence_score": round(confidence, 2), # "confidence = max(p_ai, 1 - p_ai)"
            "ai_probability": round(p_ai, 2),
            "detected_language": detected_language,
            "transcription": transcription,
            "english_translation": english_translation,
            "fraud_keywords": found_keywords,
            "overall_risk": overall_risk,
            "explanation": explanation,
            # Diagnostic info
            "audio_duration_seconds": audio_duration_seconds,
            "num_chunks_processed": len(chunks),
            "chunk_ai_probabilities": [round(p, 3) for p in ai_probs],
            # Deep diagnostics
            "heuristic_score": round(heuristic_score, 3),
            "pitch_human_score": round(pitch_score, 3),
            "pitch_std": round(p_std, 2),
            "pitch_jitter": round(p_jitter, 2),
            "smoothness_score": round(smoothness, 4),
            "variance_score": round(time_variance, 5),
            "snr_score": round(snr_val, 2) if 'snr_val' in locals() else 0.0,
            "debug_probs": [round(p, 4) for p in probs[0].tolist()] if probs is not None else [],
            "debug_labels": self.model.config.id2label if self.model.config.id2label else "None"
        }

# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = KeywordOnlyDetector() if ML_DISABLED else VoiceDetector.get_instance()
    return detector


class KeywordOnlyDetector:
    """
    Lightweight detector for local debugging / environments where torch can't be imported.
    This keeps `/detect` alive so you can test auth/validation/Postman without ML inference.
    """

    def __init__(self):
        # Minimal subset required by routes.py.
        # NOTE: Keep in sync with VoiceDetector.__init__ keyword definitions.
        self.risk_categories = {
            "secrets": [
                "otp", "one time password", "tell me otp", "share otp", "send otp",
                "cvv", "pin", "atm pin", "upi pin",
                "password", "netbanking", "credentials",
                "screen share", "anydesk", "teamviewer",
                "sms code", "verification code", "tell the code",
            ],
            "threats": [
                "account will be blocked", "account blocked", "sim will be deactivated",
                "final notice", "last chance", "immediately", "urgent",
                "before lines close", "within 1 hour", "within 2 hours",
                "within 24 hours", "within 23 hours",
            ],
            "prizes": [
                "you have won", "you won", "congratulations you are selected",
                "congratulations", "claim your prize", "prize", "cash reward",
                "reward", "lottery", "lucky draw", "bonus caller",
                "free holiday", "voucher",
            ],
            "payments": [
                "pay now", "pay on this", "send money", "transfer money",
                "processing fee", "logistics charge", "deposit to receive",
            ],
            "premium": [
                "premium", "landline", "charged per", "10ppm", "150p/min", "090", "087"
            ],
            "institutions": [
                "bank", "sbi", "hdfc", "icici", "axis", "union bank",
                "customer care", "manager", "verification", "statement", "kyc",
                "bhim", "upi upgrade", "account"
            ],
            "cta": [
                "call now", "call this number", "call immediately",
                "click link", "click here", "visit website",
            ],
            "generic": [
                "suspicious", "security", "verify", "identity", "activity"
            ]
        }

        self.multilingual_high = {
            "hindi_devanagari": ["ओटीपी", "पिन", "पासवर्ड", "ब्लॉक", "तुरंत", "लॉटरी"],
            "hindi_roman": ["otp", "pin", "band", "jald", "jaldi"],
            "tamil_native": ["ஓடீபி", "பின்", "பாஸ்வர்ட்", "அவசரம்", "பரிசு", "ஒடிபி"],
            "telugu_native": ["ఓటీపీ", "పిన్", "పాస్వర్డ్", "అత్యవసరం", "బహుమతి"],
            "malayalam_native": ["ഒടിപി", "പിൻ", "പാസ്‌വേർഡ്", "അടിയന്തിരം", "സമ്‌മാനം"],
        }

        self.high_risk_keywords = {
            "english": (
                self.risk_categories["secrets"] +
                self.risk_categories["threats"] +
                self.risk_categories["prizes"] +
                self.risk_categories["payments"] +
                self.risk_categories["premium"]
            ),
            **self.multilingual_high
        }

        self.low_risk_keywords = {
            "english": (
                self.risk_categories["institutions"] + 
                self.risk_categories["cta"] + 
                self.risk_categories["generic"]
            ),
            "hindi_devanagari": ["बैंक", "खाता", "वेरिफाई", "नंबर", "कार्ड"],
            "hindi_roman": ["bank", "khata", "verify", "number"],
            # Include common Tamil-script loanwords used in scam calls.
            "tamil_native": ["கணக்கு", "வங்கி", "பேங்க்", "அக்கவுண்ட்", "வெரிஃபை", "கஸ்டமர்", "கஸ்டமர் கேர்", "கஸ்டமர் கேர்", "அன்ஆத்தரைஸ்டா"],
            "telugu_native": ["ఖాతా", "బ్యాం‌క్"],
            "malayalam_native": ["അക്കൗണ്ട്", "ബാങ്ക്"],
        }

    def _check_keywords(self, transcript: str):
        # Copied logic from VoiceDetector._check_keywords
        import re

        if not transcript:
            return [], "unknown", 0, 0

        transcript_lower = transcript.lower()
        found = []
        high_count = 0
        low_count = 0
        seen = set()

        def _use_word_boundaries(kw: str) -> bool:
            return kw.isascii()

        for lang, keywords in self.high_risk_keywords.items():
            for kw in keywords:
                if _use_word_boundaries(kw):
                    pattern = r"\\b" + re.escape(kw) + r"\\b"
                    hit = re.search(pattern, transcript_lower) is not None
                else:
                    hit = kw in transcript_lower
                if hit:
                    tag = f"{kw} ({lang}) [HIGH]"
                    if tag not in seen:
                        found.append(tag)
                        seen.add(tag)
                        high_count += 1

        for lang, keywords in self.low_risk_keywords.items():
            for kw in keywords:
                if _use_word_boundaries(kw):
                    pattern = r"\\b" + re.escape(kw) + r"\\b"
                    hit = re.search(pattern, transcript_lower) is not None
                else:
                    hit = kw in transcript_lower
                if hit:
                    tag = f"{kw} ({lang}) [LOW]"
                    if tag not in seen:
                        found.append(tag)
                        seen.add(tag)
                        low_count += 1

        return found, "multilingual", high_count, low_count

    def detect_fraud(self, input_audio, metadata=None, transcript=None):
        try:
            audio_len = len(input_audio) if hasattr(input_audio, "__len__") else 0
        except Exception:
            audio_len = 0

        audio_duration_seconds = round(audio_len / 16000, 2) if audio_len else 0.0

        return {
            "classification": "Human",
            "confidence_score": 0.5,
            "ai_probability": 0.0,
            "detected_language": "unknown",
            "transcription": transcript or "",
            "english_translation": "",
            "fraud_keywords": [],
            "overall_risk": "LOW",
            "explanation": "ML disabled (DISABLE_ML=1): keyword-only mode",
            "audio_duration_seconds": audio_duration_seconds,
            "pitch_human_score": 0.0,
            "metadata_flag": None,
        }
