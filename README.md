# Ski-Bot
AI-powered skin and eye screening — describe your symptoms by voice or text, upload a photo, and get an instant AI-generated analysis with care guidance.

# What it does
🩹 Skin screening — classifies photos across 23 skin conditions (acne, eczema, psoriasis, melanoma, and more)
👁️ Eye screening — classifies eye photos across cataract, diabetic retinopathy, glaucoma, and normal
🎙️ Voice input — record your symptoms; Gemini transcribes them automatically
🤖 AI-generated guidance — Gemini explains the predicted condition and gives general self-care tips
⚠️ Always recommends seeing a licensed specialist — this is a screening aid, not a diagnostic tool

# Datasets used
Skin: DermNet — 23 skin disease categories
Eye: Eye Diseases Classification — cataract, diabetic retinopathy, glaucoma, normal

# Project structure
├── app.py                             # Flask backend — model inference + Gemini integration
├── skibot.html                        # Frontend UI (served by Flask at "/")
├── skin_disease_classifier_colab.py   # Skin model training notebook (run in Google Colab)
├── eye_disease_classifier_colab.py    # Eye model training notebook (run in Google Colab)
├── skin_diseases_final.keras          # Trained skin model
├── eye_disease_model_final.keras      # Trained eye model
├── requirements.txt                   # Python dependencies
├── .env                                # Local secrets (GEMINI_API_KEY) — not committed
└── .gitignore

# Known limitations
Model confidence varies with image quality, lighting, and how closely the photo matches training data
Voice transcription and remedy generation depend on the Gemini API being reachable
No user accounts, saved history, or PDF export yet
This is a personal/educational project, not a clinically validated medical device
