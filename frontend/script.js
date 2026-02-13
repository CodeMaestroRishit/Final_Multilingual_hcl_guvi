// ===== CONFIG =====
const API_URL = "https://multilingual-guvi-hcl-production.up.railway.app/detect";

// ===== DOM ELEMENTS =====
const uploadCard = document.getElementById("uploadCard");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const clearBtn = document.getElementById("clearBtn");
const apiKeyInput = document.getElementById("apiKey");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultsSection = document.getElementById("resultsSection");
const errorCard = document.getElementById("errorCard");
const errorText = document.getElementById("errorText");

// ===== SERVER WARMUP =====
// Ping the server immediately to wake it up from cold start
const SERVER_STATUS_URL = "https://multilingual-guvi-hcl-production.up.railway.app/ping";
const statusBadge = document.createElement("div");
statusBadge.className = "status-badge";
statusBadge.innerText = "🟠 Connecting to Neural Engine...";
document.body.appendChild(statusBadge);

fetch(SERVER_STATUS_URL)
    .then(res => {
        if (res.ok) {
            statusBadge.innerText = "🟢 System Ready";
            statusBadge.classList.add("ready");
            setTimeout(() => statusBadge.style.opacity = "0", 3000); // Fade out after 3s
        }
    })
    .catch(err => {
        console.log("Warmup ping failed:", err);
        statusBadge.innerText = "🔴 Server Offline (Check Connection)";
    });

let selectedFile = null;

// ===== FILE UPLOAD =====
uploadCard.addEventListener("click", () => fileInput.click());

uploadCard.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadCard.classList.add("drag-over");
});

uploadCard.addEventListener("dragleave", () => {
    uploadCard.classList.remove("drag-over");
});

uploadCard.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadCard.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("audio/")) {
        handleFile(file);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.style.display = "none";
    analyzeBtn.disabled = true;
    resultsSection.style.display = "none";
    errorCard.style.display = "none";
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    fileInfo.style.display = "flex";
    analyzeBtn.disabled = false;
    resultsSection.style.display = "none";
    errorCard.style.display = "none";
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

// ===== ANALYZE =====
analyzeBtn.addEventListener("click", analyze);

async function analyze() {
    if (!selectedFile) return;

    const btnText = analyzeBtn.querySelector(".btn-text");
    const btnLoader = analyzeBtn.querySelector(".btn-loader");

    // UI: Loading state
    btnText.style.display = "none";
    btnLoader.style.display = "inline";
    analyzeBtn.disabled = true;
    resultsSection.style.display = "none";
    errorCard.style.display = "none";

    try {
        // Convert file to base64
        const base64 = await fileToBase64(selectedFile);
        const apiKey = apiKeyInput.value.trim();
        if (!apiKey) {
            throw new Error("Please enter your API key");
        }

        const startTime = performance.now();

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-api-key": apiKey
            },
            body: JSON.stringify({ audio_base64: base64 })
        });

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || `API Error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data, elapsed);

    } catch (err) {
        errorText.textContent = err.message;
        errorCard.style.display = "flex";
    } finally {
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
        analyzeBtn.disabled = false;
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const base64 = reader.result.split(",")[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ===== RENDER RESULTS =====
function renderResults(data, elapsed) {
    const verdictCard = document.getElementById("verdictCard");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictText = document.getElementById("verdictText");

    // Extract from nested response
    const voice = data.voice || {};
    const fraud = data.fraud || {};
    const transcript = data.transcript || {};
    const diag = data.diagnostics || {};

    const isFraud = fraud.fraud_detected;
    const risk = fraud.risk_level || "LOW";
    const isAI = voice.classification === "AI";

    let verdictClass, icon, text;

    if (isFraud && risk === "HIGH") {
        verdictClass = "danger";
        icon = "🚨";
        text = "Fraud Detected";
    } else if (risk === "MEDIUM" || isAI) {
        verdictClass = "medium";
        icon = "⚠️";
        text = isAI ? "AI Voice Detected" : "Suspicious Activity";
    } else {
        verdictClass = "safe";
        icon = "✅";
        text = "Safe Call";
    }

    verdictCard.className = "verdict-card " + verdictClass;
    verdictIcon.textContent = icon;
    verdictText.textContent = text;

    // Stats
    document.getElementById("statClassification").textContent = voice.classification || "—";
    document.getElementById("statClassification").style.color =
        isAI ? "var(--red)" : "var(--green)";

    document.getElementById("statRisk").textContent = risk;
    document.getElementById("statRisk").style.color =
        risk === "HIGH" ? "var(--red)" : risk === "MEDIUM" ? "var(--yellow)" : "var(--green)";

    document.getElementById("statConfidence").textContent =
        ((voice.confidence || 0) * 100).toFixed(0) + "%";

    const langMap = {
        "en-IN": "English 🇮🇳", "hi-IN": "Hindi 🇮🇳", "ta-IN": "Tamil 🇮🇳",
        "te-IN": "Telugu 🇮🇳", "ml-IN": "Malayalam 🇮🇳", "kn-IN": "Kannada 🇮🇳",
        "bn-IN": "Bengali 🇮🇳", "mr-IN": "Marathi 🇮🇳", "gu-IN": "Gujarati 🇮🇳",
        "unknown": "Unknown"
    };
    document.getElementById("statLanguage").textContent =
        langMap[transcript.language] || transcript.language || "—";

    // AI Probability Gauge
    const pct = Math.round((voice.ai_probability || 0) * 100);
    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeValue = document.getElementById("gaugeValue");

    gaugeValue.textContent = pct + "%";
    setTimeout(() => {
        gaugeFill.style.width = pct + "%";
        if (pct > 60) {
            gaugeFill.style.background = "linear-gradient(90deg, var(--yellow), var(--red))";
        } else if (pct > 35) {
            gaugeFill.style.background = "linear-gradient(90deg, var(--green), var(--yellow))";
        } else {
            gaugeFill.style.background = "linear-gradient(90deg, var(--green), #4caf50)";
        }
    }, 100);

    // Transcription
    const transcriptCard = document.getElementById("transcriptCard");
    if (transcript.original) {
        transcriptCard.style.display = "block";
        document.getElementById("transcriptOriginal").textContent = transcript.original;

        const translationDiv = document.getElementById("transcriptTranslation");
        if (transcript.english) {
            translationDiv.style.display = "block";
            document.getElementById("translationText").textContent = transcript.english;
        } else {
            translationDiv.style.display = "none";
        }
    } else {
        transcriptCard.style.display = "none";
    }

    // Keywords
    const keywordsCard = document.getElementById("keywordsCard");
    const keywordsList = document.getElementById("keywordsList");
    const keywords = fraud.keywords_found || [];
    if (keywords.length > 0) {
        keywordsCard.style.display = "block";
        keywordsList.innerHTML = keywords.map(kw => {
            const isHigh = kw.includes("[HIGH]");
            const cls = isHigh ? "high" : "low";
            const dot = isHigh ? "🔴" : "🟡";
            return `<span class="keyword-tag ${cls}">${dot} ${kw}</span>`;
        }).join("");
    } else {
        keywordsCard.style.display = "none";
    }

    // Risk Reasons (new!)
    const explanationEl = document.getElementById("explanationText");
    let explanationStr = data.explanation || "";
    if (fraud.risk_reasons && fraud.risk_reasons.length > 0) {
        explanationStr += " | Reasons: " + fraud.risk_reasons.join("; ");
    }
    explanationStr += ` | Processed in ${elapsed}s`;
    explanationEl.textContent = explanationStr;

    // Show results
    resultsSection.style.display = "flex";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}
