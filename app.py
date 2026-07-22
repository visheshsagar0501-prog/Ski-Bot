"""
=============================================================
  FLASK BACKEND API — DermaBot
  Skin Disease + Eye Disease Classifier
  + Gemini-powered Remedies + Voice Transcription
=============================================================

ENDPOINTS:
  GET  /health        — confirm both models loaded correctly
  POST /predict       — skin disease prediction
  POST /predict-eye   — eye disease prediction
  POST /transcribe    — voice to text (used before analysis)

INSTALL:
  pip install flask flask-cors tensorflow numpy pillow requests

SET YOUR GEMINI KEY (never hardcode it):
  Mac/Linux : export GEMINI_API_KEY="your-key-here"
  Windows   : $env:GEMINI_API_KEY="your-key-here"

RUN:
  python app.py
=============================================================
"""

import os
import io
import base64
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
from pydub import AudioSegment
import imageio_ffmpeg

# pydub needs an ffmpeg binary — imageio_ffmpeg bundles one, no system install needed
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


# ─────────────────────────────────────────────
# SKIN MODEL CONFIGURATION
# ─────────────────────────────────────────────

SKIN_MODEL_PATH = "skin_diseases_final.keras"

SKIN_CLASS_NAMES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

SKIN_IMG_SIZE = (160, 160)


# ─────────────────────────────────────────────
# EYE MODEL CONFIGURATION
# ─────────────────────────────────────────────

EYE_MODEL_PATH = "eye_disease_model_final.keras"

# ⚠️ Must match EXACTLY what Cell 5 printed in your eye Colab notebook
EYE_CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

EYE_IMG_SIZE = (160, 160)


# ─────────────────────────────────────────────
# GEMINI CONFIGURATION
# Read from environment variable — never hardcode your key
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "Run: export GEMINI_API_KEY='your-key-here'  (Mac/Linux)\n"
        "  or: $env:GEMINI_API_KEY='your-key-here'   (Windows PowerShell)"
    )

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ─────────────────────────────────────────────
# INITIALIZE FLASK APP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
# LOAD BOTH MODELS AT STARTUP
# ─────────────────────────────────────────────

print("📦 Loading skin model...")
if not os.path.exists(SKIN_MODEL_PATH):
    raise FileNotFoundError(f"Skin model not found at '{SKIN_MODEL_PATH}'.")
skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH)
print("✅ Skin model loaded.")

print("📦 Loading eye model...")
if not os.path.exists(EYE_MODEL_PATH):
    raise FileNotFoundError(f"Eye model not found at '{EYE_MODEL_PATH}'.")
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
print("✅ Eye model loaded.")

print("\n🚀 DermaBot backend is ready — both models online.\n")


# ─────────────────────────────────────────────
# HELPER: PREPROCESS IMAGE
# ─────────────────────────────────────────────

def preprocess_image(image_bytes, img_size):
    """
    Resizes and converts image bytes into the numpy array
    shape each model expects: (1, H, W, 3)
    img_size must match what that specific model was trained with.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ─────────────────────────────────────────────
# HELPER: CALL GEMINI
# ─────────────────────────────────────────────

def call_gemini(parts):
    """
    Sends a multimodal request to Gemini.
    'parts' can contain text and/or inline base64 media.
    Returns Gemini's text response.
    """
    response = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": parts}]},
        timeout=30
    )
    if not response.ok:
        # Raise a clean error WITHOUT the URL (which contains the API key)
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:300]}")
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─────────────────────────────────────────────
# HELPER: TRANSCRIBE VOICE AUDIO
# ─────────────────────────────────────────────

def convert_to_wav(audio_bytes, mime_type):
    """
    Browsers record audio as webm/mp4/ogg depending on the device —
    but Gemini only officially supports WAV, MP3, AIFF, AAC, OGG, and FLAC.
    Sending webm directly causes a 400 error from Gemini's API.
    This converts whatever the browser sent into WAV, which Gemini always accepts.
    """
    if "mp4" in mime_type:
        src_format = "mp4"
    elif "ogg" in mime_type:
        src_format = "ogg"
    elif "wav" in mime_type:
        src_format = "wav"
    else:
        src_format = "webm"

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=src_format)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()


def transcribe_audio(audio_bytes, mime_type="audio/webm"):
    """
    Converts the recorded audio to WAV, then sends it to Gemini for transcription.
    Returns the transcribed text string, or "" on failure.
    """
    try:
        wav_bytes = convert_to_wav(audio_bytes, mime_type)
    except Exception as e:
        print(f"⚠️ Audio conversion failed: {e}")
        return ""

    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    parts = [
        {"text": "Transcribe exactly what the person says in this audio "
                 "about their symptoms. Return ONLY the transcribed text, nothing else."},
        {"inline_data": {"mime_type": "audio/wav", "data": audio_base64}}
    ]
    try:
        return call_gemini(parts)
    except Exception as e:
        print(f"⚠️ Audio transcription failed: {e}")
        return ""


# ─────────────────────────────────────────────
# HELPER: GET REMEDY FROM GEMINI
# Works for both skin and eye conditions
# ─────────────────────────────────────────────

def get_remedy(disease_name, symptom_text="", specialist="dermatologist"):
    """
    Asks Gemini for calm, general care guidance.
    'specialist' changes the recommendation at the end
    (e.g. 'dermatologist' for skin, 'ophthalmologist' for eyes).
    """
    prompt = f"""A medical image classification AI predicted the condition: "{disease_name}".
The patient described their symptoms as: "{symptom_text or 'No additional description provided.'}"

Provide:
1. A brief, general explanation of this condition (2 sentences)
2. 3-4 general self-care tips (no specific drug names or dosages)
3. A clear reminder to see a licensed {specialist} for proper diagnosis and treatment

Keep the tone calm and reassuring. Do not provide a definitive diagnosis or prescribe medication."""

    try:
        return call_gemini([{"text": prompt}])
    except Exception as e:
        print(f"⚠️ Gemini remedy call failed: {e}")
        return (
            "Could not fetch detailed guidance right now. "
            f"Please consult a {specialist} for proper care."
        )


# ─────────────────────────────────────────────
# ROUTE: SERVE FRONTEND
# ─────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "skibot.html")


# ─────────────────────────────────────────────
# ROUTE: HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "skin_model_loaded": True,
        "eye_model_loaded": True
    })


# ─────────────────────────────────────────────
# ROUTE: TRANSCRIBE (voice → text, live, before analysis)
# ─────────────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    mime_type = audio_file.mimetype or "audio/webm"
    text = transcribe_audio(audio_bytes, mime_type)
    return jsonify({"text": text})


# ─────────────────────────────────────────────
# ROUTE: PREDICT SKIN
# POST /predict
# Form fields: image (required), audio (optional), symptoms (optional)
# Returns: disease, confidence, symptom_description, remedy, disclaimer
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400

    image_file = request.files["image"]
    symptom_text = request.form.get("symptoms", "").strip()

    if not symptom_text and "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype or "audio/webm"
        symptom_text = transcribe_audio(audio_bytes, mime_type)

    try:
        image_bytes = image_file.read()
        img_array = preprocess_image(image_bytes, SKIN_IMG_SIZE)

        predictions = skin_model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        disease = SKIN_CLASS_NAMES[predicted_index]
        confidence = round(100 * float(np.max(predictions[0])), 2)

        remedy = get_remedy(disease, symptom_text, specialist="dermatologist")

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "symptom_description": symptom_text,
            "remedy": remedy,
            "disclaimer": "This is an AI estimate, not a medical diagnosis. Please consult a licensed dermatologist."
        })

    except Exception as e:
        return jsonify({"error": f"Skin prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: PREDICT EYE
# POST /predict-eye
# Form fields: image (required), audio (optional), symptoms (optional)
# Returns: disease, confidence, symptom_description, remedy, disclaimer
# ─────────────────────────────────────────────

@app.route("/predict-eye", methods=["POST"])
def predict_eye():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400

    image_file = request.files["image"]
    symptom_text = request.form.get("symptoms", "").strip()

    if not symptom_text and "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype or "audio/webm"
        symptom_text = transcribe_audio(audio_bytes, mime_type)

    try:
        image_bytes = image_file.read()
        img_array = preprocess_image(image_bytes, EYE_IMG_SIZE)

        predictions = eye_model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        disease = EYE_CLASS_NAMES[predicted_index]
        confidence = round(100 * float(np.max(predictions[0])), 2)

        remedy = get_remedy(disease, symptom_text, specialist="ophthalmologist")

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "symptom_description": symptom_text,
            "remedy": remedy,
            "disclaimer": "This is an AI estimate, not a medical diagnosis. Please consult a licensed ophthalmologist."
        })

    except Exception as e:
        return jsonify({"error": f"Eye prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# RUN THE SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
