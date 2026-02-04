from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
from pydub import AudioSegment
import librosa
import numpy as np

app = FastAPI()

SECRET_API_KEY = "sk_test_123456789"
ALLOWED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


def decode_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)

    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3.write(audio_bytes)
    temp_mp3.close()

    audio = AudioSegment.from_mp3(temp_mp3.name)
    audio = audio.set_channels(1).set_frame_rate(22050)

    samples = np.array(audio.get_array_of_samples()).astype(float)
    return samples


def extract_features(samples):
    mfcc = librosa.feature.mfcc(y=samples, sr=22050, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(samples))
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=samples, sr=22050)
    )

    features = np.hstack([mfcc_mean, zcr, spectral_centroid])
    return features


def classify_voice(features):
    smoothness = np.var(features)

    if smoothness < 50:
        return "AI_GENERATED", 0.85, "Unnatural smoothness detected"
    else:
        return "HUMAN", 0.80, "Natural voice variations detected"


@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if data.language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    samples = decode_audio(data.audioBase64)
    features = extract_features(samples)

    classification, confidence, explanation = classify_voice(features)

    return {
        "status": "success",
        "language": data.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
