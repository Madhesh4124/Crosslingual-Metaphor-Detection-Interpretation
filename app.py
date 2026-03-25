import os
import torch
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
from google import genai
from dotenv import load_dotenv

# Load configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MONGODB_URL = os.environ.get("MONGODB_URL")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "metaphor_detector")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 7860))

app = FastAPI(title="Multilingual Metaphor Detector")

# Enable CORS for Vercel and all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
db_client: Optional[AsyncIOMotorClient] = None
models = {}
tokenizers = {}

# Language Mapping
SUPPORTED_LANGUAGES = ['hindi', 'tamil', 'telugu', 'kannada']

class DetectRequest(BaseModel):
    text: str
    language: Optional[str] = None # Optional: If not provided, we detect it.

# Database Connection
async def connect_db():
    global db_client
    try:
        if not MONGODB_URL:
            logger.error("MONGODB_URL missing!")
            return None
        db_client = AsyncIOMotorClient(MONGODB_URL)
        await db_client.admin.command('ping')
        logger.info("✓ Connected to MongoDB Atlas")
        return db_client[MONGODB_DB_NAME]
    except Exception as e:
        logger.error(f"✗ MongoDB Connection Failed: {e}")
        return None

# Model Repo IDs on Hugging Face Hub
MODEL_REPO_IDS = {
    "tamil": "Madhesh4124/tamil-metaphor-xlm",
    "telugu": "Madhesh4124/telugu-metaphor-muril",
    "kannada": "Madhesh4124/kannada-metaphor-bert",
    "hindi": "Madhesh4124/hindi-metaphor-xlm"
}

# Efficient Model Loading from Hugging Face Hub
def load_models():
    """Load from the Hub Repo IDs you've already pushed"""
    for lang, repo_id in MODEL_REPO_IDS.items():
        try:
            logger.info(f"Downloading/Loading {lang} from Hub: {repo_id}...")
            tokenizers[lang] = AutoTokenizer.from_pretrained(repo_id)
            models[lang] = AutoModelForSequenceClassification.from_pretrained(
                repo_id,
                torch_dtype=torch.float16, # Vital to fit all 4 in 16GB
                low_cpu_mem_usage=True
            )
            models[lang].eval()
            logger.info(f"✓ {lang} model ready.")
        except Exception as e:
            logger.error(f"Failed to load {lang} from Hub: {e}")
    
    if not models:
        logger.warning("No models could be loaded from Hugging Face Hub.")

@app.on_event("startup")
async def startup_event():
    await connect_db()
    load_models()

@app.get("/")
async def root():
    return {"message": "Metaphor Detection API is running on Hugging Face Spaces"}

@app.get("/health")
async def health():
    db_status = "connected" if db_client else "disconnected"
    loaded_models = list(models.keys())
    return {
        "status": "online",
        "database": db_status,
        "models_loaded": loaded_models,
        "ram_available": "16GB Tier"
    }

@app.post("/detect")
async def detect_metaphor(request: DetectRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1. Language Detection
    lang = request.language if request.language in SUPPORTED_LANGUAGES else None
    if not lang:
        try:
            detected = detect(text)
            lang_map = {'hi': 'hindi', 'ta': 'tamil', 'te': 'telugu', 'kn': 'kannada'}
            lang = lang_map.get(detected, 'hindi')
        except:
            lang = 'hindi' # Default to Hindi

    # 2. Prediction
    if lang not in models:
        # Fallback to Hindi if requested language model isn't loaded
        if 'hindi' in models:
            lang = 'hindi'
        else:
            raise HTTPException(status_code=500, detail=f"No models available for prediction.")

    tokenizer = tokenizers[lang]
    model = models[lang]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    result = {
        "text": text,
        "language": lang,
        "label": "metaphor" if pred_class == 1 else "normal",
        "confidence": float(confidence),
        "timestamp": datetime.now().isoformat()
    }

    # 3. Save to MongoDB
    if db_client:
        try:
            db = db_client[MONGODB_DB_NAME]
            await db.predictions.insert_one(result.copy())
        except Exception as e:
            logger.error(f"Database save error: {e}")

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
