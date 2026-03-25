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

# Enable CORS for all origins
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

# Model Repo IDs on Hugging Face Hub
MODEL_REPO_IDS = {
    "tamil": "Madhesh4124/tamil-metaphor-xlm",
    "telugu": "Madhesh4124/telugu-metaphor-muril",
    "kannada": "Madhesh4124/kannada-metaphor-bert",
    "hindi": "Madhesh4124/hindi-metaphor-xlm"
}

class DetectRequest(BaseModel):
    text: str
    language: Optional[str] = None

# Database Connection
async def connect_db():
    global db_client
    try:
        if not MONGODB_URL:
            logger.error("!!! MONGODB_URL IS MISSING !!!")
            return None
        
        logger.info("📡 Attempting to connect to MongoDB...")
        # Add a short timeout so it doesn't hang forever
        db_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000) 
        
        # This line is usually where it hangs if the IP isn't whitelisted
        await asyncio.wait_for(db_client.admin.command('ping'), timeout=5.0)
        
        logger.info("✓ Connected to MongoDB Atlas")
        return db_client[MONGODB_DB_NAME]
    except asyncio.TimeoutError:
        logger.error("✗ MongoDB Connection Timed Out. Check your IP Whitelist!")
        return None
    except Exception as e:
        logger.error(f"✗ MongoDB Connection Failed: {e}")
        return None

# Lazy Model Loading Function
def get_model(lang: str):
    """Load model on demand (Lazy Loading) from Hugging Face Hub"""
    if lang not in models:
        repo_id = MODEL_REPO_IDS.get(lang)
        if not repo_id:
            logger.warning(f"Language {lang} not supported. Defaulting to hindi.")
            lang = 'hindi'
            repo_id = MODEL_REPO_IDS['hindi']
            
        if lang not in models:
            logger.info(f"⏳ LAZY LOADING {lang} model from Hub: {repo_id}...")
            try:
                tokenizers[lang] = AutoTokenizer.from_pretrained(repo_id)
                models[lang] = AutoModelForSequenceClassification.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                models[lang].eval()
                logger.info(f"✓ {lang} model ready.")
            except Exception as e:
                logger.error(f"Failed to lazy load {lang}: {e}")
                raise HTTPException(status_code=500, detail=f"Model loading failed.")
    
    return models[lang], tokenizers[lang], lang

@app.on_event("startup")
async def startup_event():
    await connect_db()
    logger.info("✓ Application started (Lazy Loading enabled)")

@app.get("/")
async def root():
    return {"message": "Metaphor Detection API v2.1 - Build Triggered"}

@app.get("/health")
async def health():
    db_status = "connected" if db_client else "disconnected"
    return {
        "status": "online",
        "database": db_status,
        "active_models": list(models.keys())
    }

@app.post("/detect")
async def detect_metaphor(request: DetectRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1. Language Logic
    lang_req = request.language
    if lang_req not in MODEL_REPO_IDS:
        try:
            detected = detect(text)
            lang_map = {'hi': 'hindi', 'ta': 'tamil', 'te': 'telugu', 'kn': 'kannada'}
            lang_req = lang_map.get(detected, 'hindi')
        except:
            lang_req = 'hindi'

    # 2. Lazy Model Retrieval
    model, tokenizer, final_lang = get_model(lang_req)

    # 3. Inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    result = {
        "text": text,
        "language": final_lang,
        "label": "metaphor" if pred_class == 1 else "normal",
        "confidence": float(confidence),
        "timestamp": datetime.now().isoformat()
    }

    # 4. Save to MongoDB
    if db_client:
        try:
            db = db_client[MONGODB_DB_NAME]
            await db.predictions.insert_one(result.copy())
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")

    return result

# /predict alias — required by the React frontend (App.jsx calls /predict)
@app.post("/predict")
async def predict_endpoint(request: DetectRequest):
    return await detect_metaphor(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

