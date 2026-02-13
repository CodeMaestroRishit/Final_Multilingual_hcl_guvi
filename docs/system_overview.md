# Vaani Rekha (वाणी रेखा): Proactive Voice Guard

Vaani Rekha is a multi-layered security API designed to detect AI-generated voice deepfakes and social engineering scams in real-time. Unlike reactive solutions that flag numbers *after* a report is filed, Vaani Rekha provides **In-Call Prevention**.

## 🛡️ Key Competitive Differentiator: Proactive Defense
Current market leaders (Truecaller, Telco spam guards) rely primarily on **Crowdsourced Reputation** or **Post-Call Flagging**. Vaani Rekha bridges the "Detection Gap" by analyzing the **Live Audio Stream**:
*   **Intra-Call Analysis**: Detects synthetic voice artifacts and fraud intent *while the conversation is happening*.
*   **Instant Enforcement**: Delivers `AI_DETECTED` and `HIGH_RISK` warnings to the user's screen before sensitive information (OTP/Pin) is shared.
*   **Behavioral + Semantic Intelligence**: Combines how they sound (Acoustic) with what they say (Semantic) to stop fraud at the source.

## 🏗️ 3-Layer Architecture

### Layer 1: Acoustic Deepfake Detection
*   **Core Model**: `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`.
*   **Technology**: Built on **XLS-R (Wav2Vec2)**, a cross-lingual transformer-based speech representation model trained on 128 languages.
*   **Function**: Analyzes raw audio waveform patterns to distinguish between biological vocal tracts and synthetic speech generation artifacts.

### Layer 2: Forensic Audio Heuristics
*   **Metadata Inspection**: Real-time header analysis for programmatic encoder tags (e.g., `Lavf`, `FFmpeg`). Provides a deterministic signal for programmatic audio generation.
*   **Pitch Physics**: Conditional analysis of pitch jitter (micro-variations) and shimmer. Used as a tie-breaker when the deep learning model is in the "uncertain zone" (35-65% probability).

### Layer 3: Semantic Fraud Analysis
*   **Multilingual STT**: Integrated via **Sarvam AI** for high-accuracy Speech-to-Text across 10+ Indian languages (Hindi, Tamil, Telugu, Malayalam, etc.).
*   **Translation**: Automatic English translation of regional transcripts.
*   **Regex Keyword Engine**: Performs whole-word boundary matching on transcripts to identify high-risk scam triggers (OTP, Pin, Account Block, Verify) across 8 language variants.

## � Performance Metrics
*   **Acoustic Accuracy**: 96% (validated on synthetic speech benchmarks).
*   **Detection Latency**: ~1.5s for acoustic analysis; ~4s for full semantic transcription.
*   **False Positive Mitigation**: Metadata-prioritization and whole-word regex matching reduce "false flags" from legitimate AI (e.g., GPS, screen-readers).
*   **Language Coverage**: Zero-shot support for 128 languages; optimized keyword detection for top 8 Indian language variants.

## �🛠️ Deployment & Stack
*   **Framework**: FastAPI (Python)
*   **Inference**: PyTorch / Transformers
*   **Platform**: Railway (Dockerized)
*   **Key Optimizations**: Model pre-downloading in Docker layers and parallelized segment processing to keep latency under 5 seconds.

## 🚨 Risk Scoring Logic
1.  **Voice (AI/Human)**: Acoustic classification.
2.  **Risk (LOW/MED/HIGH)**: Based on semantic keywords found in the transcript.
3.  **Fraud (Boolean)**: Set to `true` ONLY if `Risk >= MEDIUM` (i.e., AI voice alone is a warning, fraud requires intent).
