import os
import re
import time
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
MONGODB_URL    = os.environ.get("MONGODB_URL")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "metaphor_detector")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 7860))

# ── Gemini Client ─────────────────────────────────────────────────────────────
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
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

# ── Global State ──────────────────────────────────────────────────────────────
db_client: Optional[AsyncIOMotorClient] = None
models    = {}
tokenizers = {}

# In-memory caches (avoids re-running model / re-calling Gemini)
_prediction_cache:    dict = {}
_interpretation_cache: dict = {}
CACHE_TTL = 3600  # 1 hour in seconds

MODEL_REPO_IDS = {
    "tamil":   "Madhesh4124/tamil-metaphor-xlm",
    "telugu":  "Madhesh4124/telugu-metaphor-muril",
    "kannada": "Madhesh4124/kannada-metaphor-bert",
    "hindi":   "Madhesh4124/hindi-metaphor-xlm"
}
LANG_MAP = {'hi': 'hindi', 'ta': 'tamil', 'te': 'telugu', 'kn': 'kannada'}
LANG_NAMES = {'hindi':'Hindi','tamil':'Tamil','telugu':'Telugu','kannada':'Kannada','english':'English'}

# ── Pydantic Models ───────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str
    language: Optional[str] = None
    interpretation_language: Optional[str] = "english"
    session_id: Optional[str] = None   # Browser UUID — isolates history per user

# ── Database ──────────────────────────────────────────────────────────────────
async def connect_db():
    global db_client
    try:
        if not MONGODB_URL:
            logger.error("!!! MONGODB_URL IS MISSING !!!")
            return None
        logger.info("📡 Connecting to MongoDB...")
        db_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        await asyncio.wait_for(db_client.admin.command('ping'), timeout=5.0)
        logger.info("✓ Connected to MongoDB Atlas")
        db = db_client[MONGODB_DB_NAME]
        await db.predictions.create_index([("timestamp", -1)])
        return db
    except asyncio.TimeoutError:
        logger.error("✗ MongoDB Timed Out — Check your IP Whitelist!")
        return None
    except Exception as e:
        logger.error(f"✗ MongoDB Failed: {e}")
        return None

# ── Lazy Model Loading ────────────────────────────────────────────────────────
def get_model(lang: str):
    if lang not in models:
        repo_id = MODEL_REPO_IDS.get(lang)
        if not repo_id:
            logger.warning(f"Language {lang} not supported. Defaulting to Hindi.")
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

# ── Inference Helper ──────────────────────────────────────────────────────────
def _infer(tokenizer, model, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
        pred  = torch.argmax(probs, dim=1).item()
        conf  = probs[0][pred].item()
    return pred, conf

# ── Gemini Interpretation ─────────────────────────────────────────────────────
def get_interpretation(text: str, language: str, target_language: str = "english") -> dict:
    empty = {"translation": "", "literal": "", "emotional": "", "philosophical": "", "cultural": ""}
    cache_key = f"{text}|{language}|{target_language}"
    if cache_key in _interpretation_cache:
        entry = _interpretation_cache[cache_key]
        if time.time() - entry['ts'] < CACHE_TTL:
            logger.info("🎯 Using cached interpretation")
            return entry['data']

    if not gemini_client:
        return {k: "⚠️ GEMINI_API_KEY not configured" for k in empty}

    try:
        src = LANG_NAMES.get(language, language.title())
        tgt = LANG_NAMES.get(target_language.lower(), target_language.title())
        prompt = f"""You are a multilingual interpretation assistant specializing in metaphor analysis.
The following sentence is written in {src}.

Sentence: "{text}"

Your task is to analyze the sentence carefully and output exactly **5 labeled lines**
in {tgt} (labels must remain in English).

Translation: Translate the sentence into natural {tgt}, preserving any metaphorical expressions. Do NOT explain or paraphrase the metaphor.
Literal: Translate the sentence word-for-word into grammatically correct {tgt}, even if the result sounds unnatural.
Emotional: Describe the emotional state or feeling conveyed by the sentence in {tgt}.
Philosophical: Explain the deeper abstract meaning or life insight behind the metaphor in {tgt}.
Cultural: Describe the Indian or general cultural understanding of this metaphor in {tgt}.

CRITICAL RULES:
- Always produce exactly 5 lines.
- Each line must begin with its label followed by a colon.
- Do NOT number, bullet, or add extra text.
- Translation and Literal must be single-sentence outputs.
- Emotional, Philosophical, and Cultural may be 1-2 sentences.
- Do not include explanations, reasoning, or commentary outside the 5 lines.

Your response must contain exactly 5 labeled lines and nothing else."""

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
                if ll.startswith('translation:'):      result['translation']    = line.split(':',1)[1].strip()
                elif ll.startswith('literal:'):        result['literal']        = line.split(':',1)[1].strip()
                elif ll.startswith('emotional:'):      result['emotional']      = line.split(':',1)[1].strip()
                elif ll.startswith('philosophical:'):  result['philosophical']  = line.split(':',1)[1].strip()
                elif ll.startswith('cultural:'):       result['cultural']       = line.split(':',1)[1].strip()

        _interpretation_cache[cache_key] = {'data': result, 'ts': time.time()}
        return result
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {k: f"⚠️ Interpretation failed: {str(e)[:80]}" for k in empty}

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    await connect_db()
    logger.info("✓ Application started (Lazy Loading enabled)")

# ── Sentence Splitter ─────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'[.?!।]+\s*', text) if s.strip()]

# ── Core Prediction Logic ─────────────────────────────────────────────────────
async def _predict(request: DetectRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long. Max 5000 chars.")

    # Check prediction cache
    interp_lang = (request.interpretation_language or "english").lower()
    cache_key   = f"{text}|{interp_lang}"
    if cache_key in _prediction_cache:
        entry = _prediction_cache[cache_key]
        if time.time() - entry['ts'] < CACHE_TTL:
            logger.info("🎯 Returning cached prediction")
            return entry['data']

    # Language auto-detect
    lang = request.language if request.language in MODEL_REPO_IDS else None
    if not lang:
        try:
            lang = LANG_MAP.get(detect(text), 'hindi')
        except:
            lang = 'hindi'

    model, tokenizer, final_lang = get_model(lang)
    sentences   = split_sentences(text)
    is_paragraph = len(sentences) > 1

    sentence_analyses = []
    metaphor_count  = 0
    normal_count    = 0
    total_confidence = 0.0
    anchor_text     = ""   # Context anchor for context-aware detection

    for sentence in sentences:
        if not sentence.strip():
            continue

        # FIRST PASS: Standard inference on sentence alone
        pred, conf = _infer(tokenizer, model, sentence)
        label = "metaphor" if pred == 1 else "normal"

        if label == "metaphor":
            # Update context anchor for next sentence
            anchor_text = sentence
        elif label == "normal" and anchor_text:
            # SECOND PASS: Context-aware check using previous metaphor as anchor
            combined = f"{anchor_text} {sentence}"
            pred_ctx, conf_ctx = _infer(tokenizer, model, combined)
            if pred_ctx == 1:
                label = "metaphor"
                conf  = conf_ctx
                logger.info(f"Context check upgraded sentence to metaphor")
            else:
                # Two consecutive normals — break the chain
                anchor_text = ""

        if label == "metaphor":
            metaphor_count += 1
        else:
            normal_count += 1
        total_confidence += conf

        # Gemini interpretation (run in thread pool to not block event loop)
        interp = await asyncio.to_thread(
            get_interpretation, sentence, final_lang, interp_lang
        )
        sentence_analyses.append({
            "sentence":        sentence,
            "label":           label,
            "confidence":      round(float(conf), 4),
            "interpretations": interp
        })

    # ── Overall label: MAJORITY wins (fixes the 3-normal/1-metaphor bug) ──────
    overall_label = "metaphor" if metaphor_count > normal_count else "normal"
    overall_conf  = total_confidence / len(sentences) if sentences else 0.0

    result = {
        "text":                   text,
        "language":               final_lang,
        "label":                  overall_label,
        "confidence":             round(float(overall_conf), 4),
        "is_paragraph":           is_paragraph,
        "sentences":              sentence_analyses,
        "interpretation_language": interp_lang,
        # Legacy fields kept for frontend compatibility
        "translation": sentence_analyses[0]["interpretations"].get("translation", "") if sentence_analyses else "",
        "timestamp":   datetime.now().isoformat()
    }

    # Cache result
    _prediction_cache[cache_key] = {'data': result, 'ts': time.time()}

    # Save to MongoDB (tagged per session)
    if db_client:
        try:
            await db_client[MONGODB_DB_NAME].predictions.insert_one({
                **result,
                "session_id": request.session_id or "anonymous"
            })
        except Exception as e:
            logger.error(f"MongoDB save error: {e}")

    return result

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Metaphor Detection API v3.0"}

@app.get("/health")
async def health():
    return {
        "status":        "online",
        "database":      "connected" if db_client else "disconnected",
        "active_models": list(models.keys()),
        "gemini":        "configured" if gemini_client else "not configured"
    }

@app.post("/predict")
async def predict(request: DetectRequest):
    return await _predict(request)

@app.post("/detect")
async def detect_endpoint(request: DetectRequest):
    return await _predict(request)

# ── History ────────────────────────────────────────────────────────────────────
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
    query = {"session_id": session_id or "anonymous"}
    if language: query["language"] = language
    if label:    query["label"]    = label
    cursor = db_client[MONGODB_DB_NAME].predictions.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    preds  = await cursor.to_list(length=limit)
    for p in preds:
        p["_id"] = str(p["_id"])
    return {"history": preds}

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str, session_id: Optional[str] = None):
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not connected")
    result = await db_client[MONGODB_DB_NAME].predictions.delete_one(
        {"_id": ObjectId(prediction_id), "session_id": session_id or "anonymous"}
    )
    return {"deleted": result.deleted_count > 0}

@app.delete("/history")
async def clear_history(session_id: Optional[str] = None):
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not connected")
    result = await db_client[MONGODB_DB_NAME].predictions.delete_many(
        {"session_id": session_id or "anonymous"}
    )
    return {"deleted_count": result.deleted_count}

@app.get("/statistics")
async def get_statistics(session_id: Optional[str] = None):
    if not db_client:
        return {"statistics": {"total_predictions": 0, "metaphor_count": 0, "normal_count": 0, "languages": {}}}
    col = db_client[MONGODB_DB_NAME].predictions
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
