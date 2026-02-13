
import aiohttp
import asyncio
import os
import logging
import base64

class SarvamClient:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.url = "https://api.sarvam.ai/speech-to-text"
        self.logger = logging.getLogger("uvicorn")

    async def detect_speech_async(self, audio_bytes: bytes, language_code: str = "unknown", timeout_seconds=4.5) -> dict:
        """
        Sends audio to Sarvam AI for Speech-to-Text.
        Returns a dict: {"transcript": str, "language": str}
        Enforces a strict timeout to ensure we don't break the global 6s limit.
        """
        empty_result = {"transcript": "", "language": "unknown"}

        if not self.api_key:
            self.logger.warning("SARVAM_API_KEY not found. Skipping STT.")
            print("⚠️  SARVAM: API key not configured. Set SARVAM_API_KEY in .env")
            return empty_result

        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        data.add_field('model', 'saaras:v3')
        
        # Language hint: "unknown" lets Sarvam auto-detect
        if language_code and language_code != "unknown":
            data.add_field('language_code', language_code)

        try:
            headers = {"api-subscription-key": self.api_key}
            print(f"🔊 SARVAM: Sending {len(audio_bytes)} bytes for STT (timeout={timeout_seconds}s)...")

            # ssl=False to handle macOS Python SSL cert issues
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.url, 
                    data=data, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcript = result.get("transcript", "")
                        language = result.get("language_code", "unknown")
                        print(f"✅ SARVAM: Got transcript ({len(transcript)} chars), language={language}")
                        print(f"   Transcript: \"{transcript[:200]}{'...' if len(transcript) > 200 else ''}\"")
                        return {"transcript": transcript, "language": language}
                    else:
                        error_text = await response.text()
                        print(f"❌ SARVAM: API Error {response.status}: {error_text}")
                        self.logger.error(f"Sarvam API Error {response.status}: {error_text}")
                        return empty_result

        except asyncio.TimeoutError:
            print(f"⏱️  SARVAM: Timed out (>{timeout_seconds}s). Proceeding without transcript.")
            self.logger.warning(f"Sarvam API Timed Out (> {timeout_seconds}s).")
            return empty_result
        except Exception as e:
            print(f"❌ SARVAM: Exception: {str(e)}")
            self.logger.error(f"Sarvam Client Exception: {str(e)}")
            return empty_result

    async def translate_text_async(self, text: str, source_lang: str = "auto", target_lang: str = "en-IN", timeout_seconds: float = 2.0) -> str:
        """
        Translates text to English using Sarvam Translate API.
        This is text-only (no audio), so it's near-instant (~100-200ms).
        Returns the translated English text, or empty string on failure.
        """
        if not self.api_key:
            return ""
        
        if not text or len(text.strip()) < 3:
            return ""
        
        # Map Sarvam STT language codes to translate API codes
        lang_map = {
            "hi-IN": "hi-IN",
            "ta-IN": "ta-IN",
            "te-IN": "te-IN",
            "ml-IN": "ml-IN",
            "bn-IN": "bn-IN",
            "kn-IN": "kn-IN",
            "mr-IN": "mr-IN",
            "gu-IN": "gu-IN",
            "en-IN": "en-IN",
            "unknown": "hi-IN",  # Default to Hindi if unknown
        }
        
        src_lang = lang_map.get(source_lang, "hi-IN")
        
        # Don't translate if already English
        if src_lang == "en-IN":
            return text
        
        payload = {
            "input": text[:1000],  # Mayura:v1 limit
            "source_language_code": src_lang,
            "target_language_code": "en-IN",
            "model": "mayura:v1",
            "mode": "formal",
            "enable_preprocessing": True,
        }
        
        try:
            headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json",
            }
            print(f"🌐 SARVAM TRANSLATE: {src_lang} → en-IN ({len(text)} chars, timeout={timeout_seconds}s)")
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    "https://api.sarvam.ai/translate",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        translated = result.get("translated_text", "")
                        print(f"✅ SARVAM TRANSLATE: Got translation ({len(translated)} chars)")
                        print(f"   English: \"{translated[:200]}{'...' if len(translated) > 200 else ''}\"")
                        return translated
                    else:
                        error_text = await response.text()
                        print(f"❌ SARVAM TRANSLATE: API Error {response.status}: {error_text}")
                        return ""
        
        except asyncio.TimeoutError:
            print(f"⏱️  SARVAM TRANSLATE: Timed out (>{timeout_seconds}s)")
            return ""
        except Exception as e:
            print(f"❌ SARVAM TRANSLATE: Exception: {str(e)}")
            return ""

# Singleton
sarvam_client = SarvamClient()
