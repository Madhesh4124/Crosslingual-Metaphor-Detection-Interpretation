import os
import re
import torch
import logging
import asyncio
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MONGODB_URL = os.environ.get("MONGODB_URL")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "metaphor_detector")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 7860))

# Gemini client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✓ Gemini client initialized")
    except Exception as e:
        logger.error(f"✗ Gemini init failed: {e}")

app = FastAPI(title="Multilingual Metaphor Detector")

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

MODEL_REPO_IDS = {
    "tamil":   "Madhesh4124/tamil-metaphor-xlm",
    "telugu":  "Madhesh4124/telugu-metaphor-muril",
    "kannada": "Madhesh4124/kannada-metaphor-bert",
    "hindi":   "Madhesh4124/hindi-metaphor-xlm"
}

# ── Pydantic Models ──────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str
    language: Optional[str] = None
    interpretation_language: Optional[str] = "english"
    session_id: Optional[str] = None  # Browser-generated UUID for history isolation

# ── Database ─────────────────────────────────────────────────────────────────
async def connect_db():
    global db_client
    try:
        if not MONGODB_URL:
            logger.error("!!! MONGODB_URL IS MISSING !!!")
            return None
        logger.info("📡 Attempting to connect to MongoDB...")
        db_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        await asyncio.wait_for(db_client.admin.command('ping'), timeout=5.0)
        logger.info("✓ Connected to MongoDB Atlas")
        db = db_client[MONGODB_DB_NAME]
        await db.predictions.create_index([("timestamp", -1)])
        return db
    except asyncio.TimeoutError:
        logger.error("✗ MongoDB Timed Out. Check your IP Whitelist!")
        return None
    except Exception as e:
        logger.error(f"✗ MongoDB Failed: {e}")
        return None

# ── Lazy Model Loading ───────────────────────────────────────────────────────
def get_model(lang: str):
    if lang not in models:
        repo_id = MODEL_REPO_IDS.get(lang)
        if not repo_id:
            lang = 'hindi'
            repo_id = MODEL_REPO_IDS['hindi']
        if lang not in models:
            logger.info(f"⏳ Lazy loading {lang} from {repo_id}...")
            try:
                tokenizers[lang] = AutoTokenizer.from_pretrained(repo_id)
                models[lang] = AutoModelForSequenceClassification.from_pretrained(
                    repo_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                models[lang].eval()
                logger.info(f"✓ {lang} model ready.")
            except Exception as e:
                logger.error(f"Failed to load {lang}: {e}")
                raise HTTPException(status_code=500, detail="Model loading failed.")
    return models[lang], tokenizers[lang], lang

# ── Gemini Interpretation ────────────────────────────────────────────────────
def get_interpretation(text: str, language: str, target_language: str = "english") -> dict:
    """Call Gemini to produce 5-layer interpretation."""
    empty = {"translation": "", "literal": "", "emotional": "", "philosophical": "", "cultural": ""}
    if not gemini_client:
        return {k: "⚠️ GEMINI_API_KEY not configured" for k in empty}
    try:
        lang_names = {'hindi':'Hindi','tamil':'Tamil','telugu':'Telugu','kannada':'Kannada','english':'English'}
        src  = lang_names.get(language, language.title())
        tgt  = lang_names.get(target_language.lower(), target_language.title())
        prompt = f"""You are a multilingual interpretation assistant specializing in metaphor analysis.
The following sentence is written in {src}.

Sentence: "{text}"

Output exactly 5 labeled lines in {tgt}. Labels must remain in English.

Translation: Natural {tgt} translation preserving metaphorical intent.
Literal: Word-for-word {tgt} translation.
Emotional: Emotional state/feeling conveyed, in {tgt}.
Philosophical: Deeper abstract meaning/life insight, in {tgt}.
Cultural: Indian or general cultural understanding, in {tgt}.

RULES: Exactly 5 lines, each starting with its label and colon. No numbering or extra text."""

        response = gemini_client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        result = dict(empty)
        if response and response.text:
            for line in response.text.split('\n'):
                line = line.strip()
                ll = line.lower()
                if ll.startswith('translation:'): result['translation'] = line.split(':',1)[1].strip()
                elif ll.startswith('literal:'):   result['literal']     = line.split(':',1)[1].strip()
                elif ll.startswith('emotional:'): result['emotional']   = line.split(':',1)[1].strip()
                elif ll.startswith('philosophical:'): result['philosophical'] = line.split(':',1)[1].strip()
                elif ll.startswith('cultural:'): result['cultural']    = line.split(':',1)[1].strip()
        return result
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {k: f"⚠️ Interpretation failed: {str(e)[:80]}" for k in empty}

# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    await connect_db()
    logger.info("✓ Application started (Lazy Loading enabled)")

# ── Utility ──────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'[.?!।]+\s*', text) if s.strip()]

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Metaphor Detection API v2.1"}

@app.get("/health")
async def health():
    return {
        "status": "online",
        "database": "connected" if db_client else "disconnected",
        "active_models": list(models.keys()),
        "gemini": "configured" if gemini_client else "not configured"
    }

async def _predict(request: DetectRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Language detection
    lang = request.language if request.language in MODEL_REPO_IDS else None
    if not lang:
        try:
            lm = {'hi':'hindi','ta':'tamil','te':'telugu','kn':'kannada'}
            lang = lm.get(detect(text), 'hindi')
        except:
            lang = 'hindi'

    interp_lang = (request.interpretation_language or "english").lower()

    # Sentence-level analysis
    sentences = split_sentences(text)
    is_paragraph = len(sentences) > 1
    sentence_analyses = []
    total_conf = 0.0
    metaphor_count = 0

    model, tokenizer, final_lang = get_model(lang)

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)
            pred  = torch.argmax(probs, dim=1).item()
            conf  = probs[0][pred].item()

        label = "metaphor" if pred == 1 else "normal"
        if label == "metaphor":
            metaphor_count += 1
        total_conf += conf

        interp = get_interpretation(sent, final_lang, interp_lang)
        sentence_analyses.append({
            "sentence":        sent,
            "label":           label,
            "confidence":      float(conf),
            "interpretations": interp
        })

    overall_label = "metaphor" if metaphor_count > 0 else "normal"
    overall_conf  = total_conf / len(sentences) if sentences else 0.0

    result = {
        "text":         text,
        "language":     final_lang,
        "label":        overall_label,
        "confidence":   float(overall_conf),
        "is_paragraph": is_paragraph,
        "sentences":    sentence_analyses,
        "timestamp":    datetime.now().isoformat()
    }

    # Save to MongoDB (tagged with session_id for isolation)
    if db_client:
        try:
            await db_client[MONGODB_DB_NAME].predictions.insert_one({
                **result,
                "interpretation_language": interp_lang,
                "session_id": request.session_id or "anonymous"
            })
        except Exception as e:
            logger.error(f"MongoDB save error: {e}")

    return result

@app.post("/predict")
async def predict(request: DetectRequest):
    return await _predict(request)

@app.post("/detect")
async def detect_endpoint(request: DetectRequest):
    return await _predict(request)

# ── History Endpoints ─────────────────────────────────────────────────────────
@app.get("/history")
async def get_history(
    limit: int = Query(50, le=100),
    skip: int = 0,
    language: Optional[str] = None,
    label: Optional[str] = None,
    session_id: Optional[str] = None
):
    if not db_client:
        return {"history": []}
    # Only return predictions for THIS session
    query = {"session_id": session_id or "anonymous"}
    if language: query["language"] = language
    if label:    query["label"]    = label
    cursor = db_client[MONGODB_DB_NAME].predictions.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    preds = await cursor.to_list(length=limit)
    for p in preds:
        p["_id"] = str(p["_id"])
    return {"history": preds}

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str, session_id: Optional[str] = None):
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not connected")
    # Only allow deleting own predictions
    result = await db_client[MONGODB_DB_NAME].predictions.delete_one(
        {"_id": ObjectId(prediction_id), "session_id": session_id or "anonymous"}
    )
    return {"deleted": result.deleted_count > 0}

@app.delete("/history")
async def clear_history():
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not connected")
    result = await db_client[MONGODB_DB_NAME].predictions.delete_many({})
    return {"deleted_count": result.deleted_count}

@app.get("/statistics")
async def get_statistics(session_id: Optional[str] = None):
    if not db_client:
        return {"statistics": {"total_predictions": 0, "metaphor_count": 0, "normal_count": 0, "languages": {}}}
    col = db_client[MONGODB_DB_NAME].predictions
    # Only count THIS session's predictions
    sid = session_id or "anonymous"
    total    = await col.count_documents({"session_id": sid})
    metaphor = await col.count_documents({"label": "metaphor", "session_id": sid})
    normal   = await col.count_documents({"label": "normal",   "session_id": sid})
    lang_data = await col.aggregate([
        {"$match": {"session_id": sid}},
        {"$group": {"_id": "$language", "count": {"$sum": 1}}}
    ]).to_list(10)
    return {"statistics": {
        "total_predictions": total,
        "metaphor_count":    metaphor,
        "normal_count":      normal,
        "languages":         {x["_id"]: x["count"] for x in lang_data}
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
