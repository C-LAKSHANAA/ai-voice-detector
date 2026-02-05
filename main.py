from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
from pydub import AudioSegment
import librosa
import numpy as np
import os

app = FastAPI()

SECRET_API_KEY = os.getenv("API_KEY", "test-key-123")

ALLOWED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# --------------------------------------------------
# Audio Decoding
# --------------------------------------------------
def decode_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)

    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3.write(audio_bytes)
    temp_mp3.close()

    audio = AudioSegment.from_mp3(temp_mp3.name)
    audio = audio.set_channels(1).set_frame_rate(22050)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    return samples


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------
def extract_features(samples):
    mfcc = librosa.feature.mfcc(y=samples, sr=22050, n_mfcc=20)
    mfcc_var = np.var(mfcc)

    zcr = np.mean(librosa.feature.zero_crossing_rate(samples))

    spectral_flatness = np.mean(
        librosa.feature.spectral_flatness(y=samples)
    )

    rms_energy = np.mean(librosa.feature.rms(y=samples))

    return mfcc_var, zcr, spectral_flatness, rms_energy


# --------------------------------------------------
# Classification Logic (FIXED)
# --------------------------------------------------
def classify_voice(mfcc_var, zcr, flatness, energy):
    """
    AI voices:
    - lower MFCC variance
    - unnaturally flat spectrum
    - consistent energy
    """

    if mfcc_var < 20 and flatness > 0.35 and energy > 0.01:
        return "AI_GENERATED", 0.78, "Synthetic smoothness and flat spectrum detected"

    return "HUMAN", 0.80, "Natural human speech variations detected"


# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if data.language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    samples = decode_audio(data.audioBase64)

    # Minimum duration check
    if len(samples) < 22050:
        raise HTTPException(
            status_code=400,
            detail="Audio too short for analysis"
        )

    mfcc_var, zcr, flatness, energy = extract_features(samples)

    classification, confidence, explanation = classify_voice(
        mfcc_var, zcr, flatness, energy
    )

    return {
        "status": "success",
        "language": data.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
