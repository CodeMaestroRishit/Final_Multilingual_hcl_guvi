from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
from app.api.auth import verify_api_key
from app.utils.audio import process_audio_input
from app.models.detector import get_detector
from app.utils.sarvam import sarvam_client
import asyncio
import io
import soundfile as sf
import base64
import time

router = APIRouter()

class DetectRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[str] = None 
    message: Optional[str] = None

# Hackathon Exact Format — Clean, Nested Response
class VoiceAnalysis(BaseModel):
    classification: str          # "AI" | "Human"
    confidence: float            # 0.0 - 1.0
    ai_probability: float        # Raw model output

class FraudAnalysis(BaseModel):
    fraud_detected: bool         # True if scam detected
    risk_level: str              # "HIGH" | "MEDIUM" | "LOW"
    risk_reasons: List[str]      # Human-readable reasons
    keywords_found: List[str]    # Detected fraud keywords

class TranscriptInfo(BaseModel):
    language: str                # e.g. "hi", "ta"
    original: str                # Native language transcript
    english: str                 # English translation

class Diagnostics(BaseModel):
    audio_duration_seconds: float
    processing_time_ms: Optional[float] = None
    pitch_human_score: Optional[float] = 0.0
    metadata_flag: Optional[str] = None

class DetectResponse(BaseModel):
    voice: VoiceAnalysis
    fraud: FraudAnalysis
    transcript: TranscriptInfo
    explanation: str             # One-line summary
    diagnostics: Diagnostics


@router.post("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: DetectRequest):
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either 'audio_base64' or 'audio_url'"
        )
    
    try:
        # 1. Load FULL audio (no duration cap) for Sarvam keyword detection
        # Scam keywords often appear later in the call (e.g., "OTP", "password"),
        # so we must NOT prune the audio for transcription.
        audio_array, metadata = process_audio_input(
            request.audio_base64, request.audio_url, max_duration=None
        )
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            raise HTTPException(status_code=400, detail="Audio decode produced no samples")
        
        total_duration = len(audio_array) / 16000
        print(f"📏 AUDIO: Loaded {total_duration:.1f}s of audio ({len(audio_array)} samples)")
        
        # 2. Parallel Execution: Voice Analysis + Multi-Segment Sarvam STT
        
        # A. MULTI-SEGMENT SARVAM: Send first 15s + last 15s in parallel
        # Scam calls often have benign openings and fraud keywords later.
        # By transcribing BOTH ends we catch keywords throughout the call.
        SEGMENT_SECONDS = 15
        segment_samples = 16000 * SEGMENT_SECONDS
        
        def make_wav(audio_slice):
            """Convert an audio slice to WAV bytes."""
            buf = io.BytesIO()
            sf.write(buf, audio_slice, 16000, format='WAV')
            return buf.getvalue()
        
        segments = []
        if len(audio_array) <= segment_samples * 2:
            # Audio is short enough — send the whole thing as one segment
            segments.append(("full", make_wav(audio_array)))
            print(f"   → Short audio ({total_duration:.1f}s): sending as single segment")
        else:
            # Send first 15s and last 15s as separate parallel requests
            first_segment = audio_array[:segment_samples]
            last_segment = audio_array[-segment_samples:]
            segments.append(("first_15s", make_wav(first_segment)))
            segments.append(("last_15s", make_wav(last_segment)))
            print(f"   → Long audio ({total_duration:.1f}s): sending FIRST 15s + LAST 15s in parallel")
        
        # B. Prepare audio for Voice Detector (Strictly < 6s to prevent timeouts)
        detector_limit = 16000 * 6
        if len(audio_array) > detector_limit:
            detector_audio = audio_array[:detector_limit]
        else:
            detector_audio = audio_array

        # Define Tasks
        detector = get_detector()
        
        # Start ALL Sarvam Tasks in parallel (IO Bound)
        start_time = time.time()
        sarvam_tasks = []
        for seg_name, seg_bytes in segments:
            task = asyncio.create_task(
                sarvam_client.detect_speech_async(seg_bytes, timeout_seconds=4.5)
            )
            sarvam_tasks.append((seg_name, task))
        
        # Run Voice Detector Task (CPU Bound) — runs concurrently with Sarvam
        loop = asyncio.get_event_loop()
        voice_result = await loop.run_in_executor(
            None, 
            detector.detect_fraud, 
            detector_audio, 
            metadata, 
            None # No transcript yet
        )
        
        # Gather ALL Sarvam results (with remaining time budget)
        transcripts = []
        sarvam_language = "unknown"
        for seg_name, task in sarvam_tasks:
            try:
                elapsed = time.time() - start_time
                remaining = 4.5 - elapsed
                
                if remaining > 0.1:
                    sarvam_result = await asyncio.wait_for(task, timeout=remaining)
                    seg_transcript = sarvam_result.get("transcript", "")
                    seg_lang = sarvam_result.get("language", "unknown")
                    if seg_transcript:
                        transcripts.append(seg_transcript)
                        sarvam_language = seg_lang  # Use last detected language
                    print(f"   ✅ Segment '{seg_name}': {len(seg_transcript)} chars, lang={seg_lang}")
                else:
                    print(f"   ⚠️  Budget exhausted. Skipping segment '{seg_name}'.")
                    task.cancel()
                    
            except asyncio.TimeoutError:
                print(f"   ⏱️  Segment '{seg_name}' timed out.")
            except Exception as e:
                print(f"   ❌ Segment '{seg_name}' failed: {e}")
        
        # Merge transcripts from all segments
        transcript = " ".join(transcripts).strip()
        
        # --- Debug: Show what Sarvam returned ---
        sarvam_elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"📊 SARVAM RESULT (total elapsed: {sarvam_elapsed:.2f}s)")
        print(f"   Segments transcribed: {len(transcripts)}/{len(segments)}")
        print(f"   Full transcript: \"{transcript[:300]}{'...' if len(transcript) > 300 else ''}\"")
        print(f"   Language:   {sarvam_language}")
        print(f"{'='*50}")
        
        # If transcript found, translate to English and run keyword checks
        if transcript:
            reasons = []
            voice_result["transcription"] = transcript
            voice_result["detected_language"] = sarvam_language
            
            # --- Translate to English (text-only, near-instant ~100-200ms) ---
            english_translation = ""
            if sarvam_language and sarvam_language != "en-IN":
                try:
                    english_translation = await asyncio.wait_for(
                        sarvam_client.translate_text_async(
                            transcript, 
                            source_lang=sarvam_language,
                            timeout_seconds=2.0
                        ),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    print("⏱️  Translation timed out")
                except Exception as e:
                    print(f"❌ Translation failed: {e}")
            
            voice_result["english_translation"] = english_translation
            
            # --- Keyword Check: Run on BOTH native transcript AND English translation ---
            # _check_keywords now returns (keywords, lang, high_count, low_count)
            native_keywords, _, native_high, native_low = detector._check_keywords(transcript)
            if english_translation:
                english_keywords, _, eng_high, eng_low = detector._check_keywords(english_translation)
            else:
                english_keywords, eng_high, eng_low = [], 0, 0
            
            # Merge and deduplicate
            all_keywords = list(set(native_keywords + english_keywords))
            total_high = native_high + eng_high
            total_low = native_low + eng_low
            voice_result["fraud_keywords"] = all_keywords
            
            print(f"🔍 KEYWORD CHECK (native):  {len(native_keywords)} kw (HIGH={native_high}, LOW={native_low})")
            print(f"🔍 KEYWORD CHECK (english): {len(english_keywords)} kw (HIGH={eng_high}, LOW={eng_low})")
            print(f"🔍 KEYWORD CHECK (merged):  {len(all_keywords)} kw (HIGH={total_high}, LOW={total_low})")
            
            # --- Rule-Based Tiered Risk Decision ---
            # Categorize keywords by their hit counts in the risk_categories
            detector_cats = detector.risk_categories
            hits = {cat: 0 for cat in detector_cats}
            found_by_cat = {cat: [] for cat in detector_cats}
            
            for kw_tag in all_keywords:
                # Format is "keyword (lang) [HIGH/LOW]" or just "keyword [HIGH/LOW]"
                kw_clean = kw_tag.split(" (")[0].split(" [")[0].lower()
                for cat, kw_list in detector_cats.items():
                    if kw_clean in kw_list:
                        hits[cat] += 1
                        found_by_cat[cat].append(kw_tag)

            risk_level = "LOW"
            reasons = []

            # 🚨 HIGH SEVERITY RULES
            # Rule 1: Secrets (OTP/PIN/CVV) -> HIGH immediately
            if hits["secrets"] > 0:
                risk_level = "HIGH"
                reasons.append("Credential harvesting (OTP/PIN/Password) requested")
            
            # Rule 2: Threat/Urgency + Action (CTA, Secrets, or Payments) -> HIGH
            if hits["threats"] > 0 and (hits["cta"] > 0 or hits["secrets"] > 0 or hits["payments"] > 0):
                risk_level = "HIGH"
                reasons.append("Coercion detected: Threat paired with immediate action demand")
            
            # Rule 3: Prize/Reward + Hooks (CTA, Secret, or Payment) -> HIGH
            if hits["prizes"] > 0 and (hits["cta"] > 0 or hits["secrets"] > 0 or hits["payments"] > 0):
                risk_level = "HIGH"
                reasons.append("Financial hook: Prize/Reward paired with suspicious action")
            
            # Rule 4: Payment demand + (Prize or Verification/Identity) -> HIGH
            if hits["payments"] > 0 and (hits["prizes"] > 0 or hits["generic"] > 0 or hits["institutions"] > 0):
                risk_level = "HIGH"
                reasons.append("Suspicious payment demand for verification or rewards")

            # Rule 5: Premium-rate patterns + (Prize or Urgency) -> HIGH
            if hits["premium"] > 0 and (hits["prizes"] > 0 or hits["threats"] > 0):
                risk_level = "HIGH"
                reasons.append("Premium-rate callback pattern detected with urgency/hook")

            # ⚠️ MEDIUM SEVERITY FALLBACK
            if risk_level != "HIGH":
                # Any high-risk category hit (Threat, Prize, Payment, Premium)
                if hits["threats"] > 0 or hits["prizes"] > 0 or hits["payments"] > 0 or hits["premium"] > 0:
                    risk_level = "MEDIUM"
                    reasons.append("Suspicious patterns detected (Threats/Prizes/Payment context)")
                # Multiple context keywords (Institutions, CTA, Generic)
                elif hits["institutions"] + hits["cta"] + hits["generic"] >= 2:
                    risk_level = "MEDIUM"
                    reasons.append("Multiple institutional/call-to-action keywords found")
                elif all_keywords:
                    risk_level = "LOW"
                    reasons.append("General banking context terms detected")

            voice_result["overall_risk"] = risk_level
            voice_result["explanation"] += f", {risk_level} RISK — " + ("; ".join(reasons) if reasons else "Keyword context: " + ", ".join(all_keywords))

        else:
            print("⚠️  No transcript from Sarvam — keyword detection skipped")
        
        # 3. Finalize Risk & Fraud Decision
        is_ai_voice = (voice_result["classification"] == "AI")
        risk_level = voice_result.get("overall_risk", "LOW")
        is_keyword_fraud = risk_level in ("HIGH", "MEDIUM")
        
        # AI voice alone bumps LOW -> MEDIUM
        if is_ai_voice and risk_level == "LOW":
            risk_level = "MEDIUM"
            reasons.append("AI-generated voice detected")
        
        if not transcript:
            reasons = ["AI-generated voice detected"] if is_ai_voice else []
        
        is_fraud = is_keyword_fraud or (is_ai_voice and risk_level != "LOW")
        
        # Build clean explanation
        explanation_parts = []
        explanation_parts.append(f"Voice classified as {voice_result['classification']} (AI probability: {voice_result['ai_probability']})")
        if voice_result.get("fraud_keywords"):
            explanation_parts.append(f"Fraud keywords detected: {', '.join(voice_result['fraud_keywords'])}")
        if risk_level == "HIGH":
            explanation_parts.append(f"HIGH RISK — {'; '.join(reasons) if reasons else 'Critical scam indicators found'}")
        elif risk_level == "MEDIUM":
            explanation_parts.append(f"MEDIUM RISK — {'; '.join(reasons) if reasons else 'Suspicious indicators detected'}")
        
        # --- Debug ---
        print(f"\n🚨 FRAUD DECISION: fraud={is_fraud}, ai_voice={is_ai_voice}, "
              f"keyword_fraud={is_keyword_fraud}, risk={risk_level}")

        # 4. Build Clean Structured Response
        processing_time = round((time.time() * 1000) - (time.time() * 1000), 1)  # placeholder
        
        return {
            "voice": {
                "classification": voice_result["classification"],
                "confidence": round(max(voice_result.get("confidence_score", 0.5), 
                                       0.95 if risk_level == "HIGH" else 0.75 if risk_level == "MEDIUM" else 0.5), 2),
                "ai_probability": voice_result["ai_probability"],
            },
            "fraud": {
                "fraud_detected": is_fraud,
                "risk_level": risk_level,
                "risk_reasons": reasons,
                "keywords_found": voice_result.get("fraud_keywords", []),
            },
            "transcript": {
                "language": voice_result.get("detected_language", "unknown"),
                "original": voice_result.get("transcription", ""),
                "english": voice_result.get("english_translation", ""),
            },
            "explanation": ". ".join(explanation_parts),
            "diagnostics": {
                "audio_duration_seconds": voice_result.get("audio_duration_seconds", 0.0),
                "processing_time_ms": processing_time,
                "pitch_human_score": voice_result.get("pitch_human_score", 0.0),
                "metadata_flag": voice_result.get("metadata_flag", None),
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_get(audio_url: str):
    """
    GET handler for Hackathon Tester. 
    Wraps the POST logic.
    """
    # Create request object
    request = DetectRequest(audio_url=audio_url)
    # Call the existing logic (we can call the service directly or the function)
    # Calling the service logic directly ensures cleaner execution stack
    return await detect_voice(request)

# --- Strict Hackathon Specification ---

class HackathonRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

class HackathonResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

from fastapi.responses import JSONResponse

@router.post("/api/voice-detection", response_model=HackathonResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice_strict(request: HackathonRequest):
    """
    Strict endpoint for Hackathon evaluation.
    Path: /api/voice-detection
    """
    # 1. format check
    if request.audioFormat.lower() != "mp3":
        return JSONResponse(
            status_code=400, 
            content={"status": "error", "message": "Only mp3 format supported"}
        )

    try:
        # 2. process audio
        # Reuse existing logic via wrapper or direct call
        # process_audio_input expects (audio_base64, audio_url)
        # It handles base64 decoding.
        
        # NOTE: process_audio_input returns (numpy array, metadata)
        # OPTIMIZATION: Decode ONLY 6 seconds max to prevent timeouts on large files
        audio_array, metadata = process_audio_input(request.audioBase64, None, max_duration=2.0)
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio decode produced no samples"},
            )
        
        # 3. Detect
        detector = get_detector()
        result = detector.detect_fraud(audio_array, metadata)
        
        # 4. Map Result
        # result: {"classification": "AI"|"Human", "confidence_score": 0.xx, "explanation": "..."}
        
        mapping = {"AI": "AI_GENERATED", "Human": "HUMAN"}
        final_class = mapping.get(result.get("classification"), "HUMAN")
        
        return {
            "status": "success",
            "language": request.language,
            "classification": final_class,
            "confidenceScore": result.get("confidence_score", 0.0),
            "explanation": result.get("explanation", "Analysis completed")
        }

    except HTTPException as he:
        # Re-wrap HTTP exceptions to strict JSON format
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )
