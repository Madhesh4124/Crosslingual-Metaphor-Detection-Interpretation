from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import numpy as np
from typing import Optional, List, Dict
import logging
import asyncio
from pathlib import Path
import json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import hashlib
import time
import re
import whisper
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential
from database import (
    connect_to_mongodb,
    close_mongodb_connection,
    save_prediction,
    get_prediction_history,
    get_prediction_by_id,
    delete_prediction,
    clear_all_history,
    get_statistics
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multilingual Metaphor Detection API")

# Simple in-memory cache for predictions
prediction_cache = {}
interpretation_cache = {}
CACHE_TTL = 3600  # 1 hour

def get_cache_key(text: str, interpretation_language: str = "english") -> str:
    """Generate cache key for text"""
    return hashlib.md5(f"{text}:{interpretation_language}".encode()).hexdigest()

def get_interpretation_cache_key(text: str, language: str, target_language: str = "english") -> str:
    """Generate cache key for interpretations"""
    return f"{language}:{target_language}:{hashlib.md5(text.encode()).hexdigest()}"

def get_cached_prediction(text: str, interpretation_language: str = "english") -> Optional[dict]:
    """Get cached prediction if available and not expired"""
    cache_key = get_cache_key(text, interpretation_language)
    if cache_key in prediction_cache:
        cached_data = prediction_cache[cache_key]
        if time.time() - cached_data['timestamp'] < CACHE_TTL:
            logger.info(f"Cache hit for text: {text[:50]}...")
            return cached_data['result']
        else:
            # Remove expired cache entry
            del prediction_cache[cache_key]
    return None

def cache_prediction(text: str, result: dict, interpretation_language: str = "english"):
    """Cache prediction result"""
    cache_key = get_cache_key(text, interpretation_language)
    prediction_cache[cache_key] = {
        'result': result,
        'timestamp': time.time()
    }
    logger.info(f"Cached prediction for text: {text[:50]}...")

# Setup Gemma API client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
    logger.warning("AI explanations will not work without a valid API key.")
    gemma_client = None
else:
    try:
        gemma_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
        logger.info("Gemma API client configured successfully")
        logger.info(f"API Key starts with: {GEMINI_API_KEY[:10]}...")
    except Exception as e:
        logger.error(f"Failed to configure Gemma API client: {str(e)}")
        GEMINI_API_KEY = None
        gemma_client = None

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
tokenizers = {}
whisper_model = None
MODEL_BASE_PATH = Path(__file__).parent.parent / "models"

# Language mapping for our supported languages
LANGUAGE_MAP = {
    'hi': 'hindi',
    'ta': 'tamil',
    'te': 'telugu',
    'kn': 'kannada'
}

class TextInput(BaseModel):
    text: str
    interpretation_language: Optional[str] = "english"

class InterpretationData(BaseModel):
    translation: str  # Direct English translation of the metaphor
    literal: str  # Literal meaning (what it says)
    emotional: str  # Emotional interpretation
    philosophical: str  # Philosophical/deeper meaning
    cultural: str  # Cultural context

class SentenceAnalysis(BaseModel):
    sentence: str
    label: str
    confidence: float
    interpretations: InterpretationData
    is_verified: Optional[bool] = None
    verification_status: Optional[str] = None
    
    class Config:
        json_encoders = {
            InterpretationData: lambda v: v.dict() if hasattr(v, 'dict') else v
        }

class PredictionResponse(BaseModel):
    language: str
    label: str
    confidence: float
    text: str
    is_paragraph: bool
    sentences: Optional[List[SentenceAnalysis]] = None
    # Verification fields
    is_verified: Optional[bool] = None
    verification_status: Optional[str] = None
    # Legacy fields for backward compatibility
    translation: Optional[str] = None
    explanation: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    source_language: str

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str

class SpeechResponse(BaseModel):
    text: str
    success: bool
    message: Optional[str] = None
    language: Optional[str] = None

def detect_language(text: str) -> str:
    """
    Detect language using langdetect library
    """
    try:
        # Use langdetect to detect language
        detected_lang = detect(text)
        logger.info(f"LangDetect result: {detected_lang}")

        # Map to our supported languages
        if detected_lang in LANGUAGE_MAP:
            return LANGUAGE_MAP[detected_lang]

        # Fallback to character-based detection for Indic languages
        # Check for Devanagari script (Hindi)
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hindi'

        # Check for Tamil script
        if any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'tamil'

        # Check for Telugu script
        if any('\u0C00' <= char <= '\u0C7F' for char in text):
            return 'telugu'

        # Check for Kannada script
        if any('\u0C80' <= char <= '\u0CFF' for char in text):
            return 'kannada'

        # Default fallback
        logger.warning(f"Unsupported language detected: {detected_lang}, defaulting to hindi")
        return 'hindi'

    except LangDetectException as e:
        logger.error(f"Language detection failed: {str(e)}")

        # Fallback to character-based detection
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hindi'
        elif any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'tamil'
        elif any('\u0C00' <= char <= '\u0C7F' for char in text):
            return 'telugu'
        elif any('\u0C80' <= char <= '\u0CFF' for char in text):
            return 'kannada'
        else:
            logger.warning("Could not detect language, defaulting to hindi")
            return 'hindi'

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation marks
    Handles both English and Indic language punctuation
    """
    # Define sentence delimiters for Indic languages and English
    # Includes: period (.), question mark (?), exclamation (!), and Devanagari danda (।)
    sentence_pattern = r'[.?!।]+\s*'
    
    # Split by sentence delimiters
    sentences = re.split(sentence_pattern, text)
    
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logger.info(f"Split text into {len(sentences)} sentence(s)")
    return sentences

def generate_gemini_interpretation(text: str, language: str, target_language: str = "English") -> InterpretationData:
    """
    Generate multi-layered interpretation using Gemini API
    Returns literal, emotional, philosophical, and cultural interpretations in the target language
    """
    # Check cache first
    cache_key = get_interpretation_cache_key(text, language, target_language)
    if cache_key in interpretation_cache:
        logger.info(f"🎯 Using cached interpretation for: {text[:30]}...")
        return interpretation_cache[cache_key]
    
    # Check if Gemini API is configured
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        return InterpretationData(
            translation="⚠️ Translation unavailable - please configure GEMINI_API_KEY",
            literal="Translation unavailable",
            emotional="AI interpretation unavailable",
            philosophical="AI interpretation unavailable",
            cultural="AI interpretation unavailable"
        )
    
    try:
        if not gemma_client:
            raise Exception("Gemma API client not configured")
        # Language name mapping
        lang_names = {
            'hindi': 'Hindi',
            'tamil': 'Tamil',
            'telugu': 'Telugu',
            'kannada': 'Kannada',
            'english': 'English'
        }
        
        language_name = lang_names.get(language, language.title())
        target_language_name = lang_names.get(target_language.lower(), target_language.title())
        
        # Enhanced prompt for multi-layered interpretation
        prompt = f"""
You are a multilingual interpretation assistant specializing in metaphor analysis.
The following sentence is written in {language_name}.

Sentence: "{text}"

Your task is to analyze the sentence carefully and output exactly **5 labeled lines**
in {target_language_name} (labels must remain in English).

Translation: Translate the sentence into natural {target_language_name}, preserving any metaphorical expressions. Do NOT explain or paraphrase the metaphor.
Literal: Translate the sentence word-for-word into grammatically correct {target_language_name}, even if the result sounds unnatural.
Emotional: Describe the emotional state or feeling conveyed by the sentence in {target_language_name}.
Philosophical: Explain the deeper abstract meaning or life insight behind the metaphor in {target_language_name}.
Cultural: Describe the Indian or general cultural understanding of this metaphor in {target_language_name}.

CRITICAL RULES:
- Always produce exactly 5 lines.
- Each line must begin with its label followed by a colon.
- Do NOT number, bullet, or add extra text.
- Translation and Literal must be single-sentence outputs.
- Emotional, Philosophical, and Cultural may be 1–2 sentences.
- If the sentence is not metaphorical:
  Translation: No metaphor detected — literal sentence. (translated to {target_language_name})
  and complete the remaining lines accordingly.
- Do not include explanations, reasoning, or commentary outside the 5 lines.

Your response must contain exactly 5 labeled lines and nothing else.
"""

        
        # Generate the interpretation with safety settings
        response = gemma_client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            interpretation_text = response.text
            logger.info(f"✅ Generated Gemini interpretation")
            logger.info(f"Raw Gemini response:\n{interpretation_text}")
            
            # Parse the response
            translation = ""
            literal = ""
            emotional = ""
            philosophical = ""
            cultural = ""
            
            lines = interpretation_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Try to extract values with case-insensitive matching
                line_lower = line.lower()
                if line_lower.startswith('translation:'):
                    translation = line.split(':', 1)[1].strip()
                    logger.info(f"Parsed translation: {translation}")
                elif line_lower.startswith('literal:'):
                    literal = line.split(':', 1)[1].strip()
                    logger.info(f"Parsed literal: {literal}")
                elif line_lower.startswith('emotional:'):
                    emotional = line.split(':', 1)[1].strip()
                    logger.info(f"Parsed emotional: {emotional}")
                elif line_lower.startswith('philosophical:'):
                    philosophical = line.split(':', 1)[1].strip()
                    logger.info(f"Parsed philosophical: {philosophical}")
                elif line_lower.startswith('cultural:'):
                    cultural = line.split(':', 1)[1].strip()
                    logger.info(f"Parsed cultural: {cultural}")
            
            # Fallback if parsing fails
            if not translation:
                logger.warning("Translation not found in parsed response, using full text as fallback")
                translation = interpretation_text[:200] if len(interpretation_text) > 200 else interpretation_text
            if not literal:
                literal = "Literal meaning available in translation"
            if not emotional:
                emotional = "Interpretation available"
            if not philosophical:
                philosophical = "Interpretation available"
            if not cultural:
                cultural = "Interpretation available"
            
            logger.info(f"Final interpretation - Translation: {translation}, Literal: {literal}")
            
            result = InterpretationData(
                translation=translation,
                literal=literal,
                emotional=emotional,
                philosophical=philosophical,
                cultural=cultural
            )
            
            # Cache the result
            interpretation_cache[cache_key] = result
            logger.info(f"💾 Cached interpretation for: {text[:30]}...")
            
            return result
        else:
            raise Exception("No response from Gemma AI")
    
    except Exception as e:
        logger.error(f"❌ Error generating Gemma interpretation: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Return error message
        return InterpretationData(
            translation=f"⚠️ Translation failed: {str(e)[:100]}",
            literal="Translation unavailable",
            emotional="AI interpretation unavailable",
            philosophical="AI interpretation unavailable",
            cultural="AI interpretation unavailable"
        )
    
def get_gemini_prediction(text: str, language: str) -> dict:
    """
    Get Gemini's prediction for whether the text contains metaphors
    Returns a dictionary with 'label' and 'confidence' keys
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        return {"label": "error", "confidence": 0.0, "error": "Gemini API key not configured"}
    
    try:
        if not gemma_client:
            raise Exception("Gemma API client not configured")
        prompt = f"""Analyze the following text and determine if it contains a metaphor. 
        Respond with only one word: 'metaphor' if it contains a metaphor, or 'normal' if it doesn't.
        
        Text: "{text}"
        
        Response (only 'metaphor' or 'normal'):"""
        response = gemma_client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        if response and hasattr(response, 'text') and response.text:
            gemma_label = response.text.strip().lower()
            # Validate response
            if gemma_label not in ['metaphor', 'normal']:
                logger.warning(f"Unexpected Gemma response: {gemma_label}")
                return {"label": "error", "confidence": 0.0, "error": "Invalid response from Gemma"}
                
            # For Gemma, we'll use a fixed high confidence since it doesn't provide probabilities
            return {"label": gemma_label, "confidence": 0.95}
            
    except Exception as e:
        logger.error(f"Error getting Gemma prediction: {str(e)}")
        
    return {"label": "error", "confidence": 0.0, "error": str(e)}

def generate_metaphor_explanation(text: str, language: str, confidence: float) -> str:
    """
    Generate contextual explanation for detected metaphors using Gemini AI
    """
    # Check if Gemini API is configured
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        return "⚠️ AI explanation unavailable - please configure GEMINI_API_KEY in .env file"

    try:
        if not gemma_client:
            raise Exception("Gemma API client not configured")
        prompt = f"""Analyze this {language} metaphor and explain it in simple English.

TEXT: "{text}"

Provide a clear explanation that:
1. Identifies what is being compared to what
2. Explains the deeper meaning or message
3. Is 1-2 sentences maximum
4. Starts with "This metaphor..." or "It compares..."

Example for "Life is a journey":
"This metaphor compares life to a journey, suggesting that life is about the experiences and growth along the way, not just reaching the end goal."

Your explanation (keep it concise and specific to this text):"""
        response = gemma_client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        if response and hasattr(response, 'text') and response.text:
            explanation = response.text.strip()

            # Clean up the response
            explanation = explanation.replace('"', '').replace('*', '').strip()
            
            # Remove any "Explanation:" prefix
            if explanation.lower().startswith('explanation:'):
                explanation = explanation[12:].strip()

            # Ensure it's not too long
            if len(explanation) > 250:
                explanation = explanation[:247] + "..."

            logger.info(f"✅ Generated AI explanation: {explanation}")
            return explanation
        else:
            raise Exception("No response from AI")

    except Exception as e:
        logger.error(f"❌ Error generating AI explanation: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Return error message so user knows to check API key
        return f"⚠️ AI explanation failed: {str(e)[:100]}. Please check your GEMINI_API_KEY in .env file."

def translate_text(text: str, source_language: str) -> str:
    """
    Translate text from source language to English with metaphor context
    Returns translated text or fallback message
    """
    try:
        logger.info(f"Translation request for {source_language}: {text}")
        
        # Manual translations for common metaphorical expressions
        metaphor_translations = {
            'hindi': {
                'दुख की चादर ने उसे ढक लिया था': 'The blanket of sorrow had covered him/her',
                'खुशी का सूरज निकला': 'The sun of happiness rose',
                'गुस्से की आग भड़की': 'The fire of anger flared up',
                'उम्मीद का दीया जला': 'The lamp of hope was lit',
                'प्रेम की नदी बह रही है': 'The river of love is flowing'
            },
            'tamil': {
                'துக்கத்தின் கடல்': 'Ocean of sorrow',
                'மகிழ்ச்சியின் சூரியன்': 'Sun of happiness',
                'கோபத்தின் நெருப்பு': 'Fire of anger'
            },
            'kannada': {
                'ದುಃಖದ ಸಮುದ್ರ': 'Ocean of sorrow',
                'ಸಂತೋಷದ ಸೂರ್ಯ': 'Sun of happiness',
                'ಕೋಪದ ಬೆಂಕಿ': 'Fire of anger'
            }
        }
        
        # Check for manual translations first
        if source_language in metaphor_translations:
            for original, translation in metaphor_translations[source_language].items():
                if original in text:
                    logger.info(f"Using manual metaphor translation: {translation}")
                    return translation
        
        # Try to use googletrans if available
        try:
            from googletrans import Translator
            translator = Translator()
            
            # Map language codes
            lang_map = {
                'hindi': 'hi',
                'tamil': 'ta',
                'kannada': 'kn',
                'telugu': 'te'
            }
            
            source_lang = lang_map.get(source_language, 'auto')
            result = translator.translate(text, src=source_lang, dest='en')
            translated_text = result.text
            
            logger.info(f"Translation successful: {translated_text}")
            return translated_text
            
        except ImportError:
            # Fallback if googletrans not installed
            logger.warning("googletrans not installed, using placeholder")
            return f"[Translation of '{text}' from {source_language} to English. Install 'googletrans==4.0.0-rc1' for automatic translation.]"
        except Exception as trans_error:
            # Fallback if translation fails
            logger.error(f"Translation error: {str(trans_error)}")
            return f"[Translation temporarily unavailable. Original text: '{text}']"
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"[Translation failed: {str(e)}]"

def load_models():
    """Load all language models at startup"""
    languages = ['hindi', 'tamil', 'telugu', 'kannada']
    loaded_count = 0
    
    for lang in languages:
        model_path = MODEL_BASE_PATH / f"{lang}_model"
        
        if not model_path.exists():
            logger.warning(f"Model path not found: {model_path}")
            continue
        
        logger.info(f"Loading {lang} model from {model_path}")
        
        try:
            # Load tokenizer with fallback to slow tokenizer
            try:
                tokenizers[lang] = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=True
                )
                logger.info(f"✓ Loaded fast tokenizer for {lang}")
            except Exception as e:
                logger.warning(f"Fast tokenizer failed for {lang}, trying slow tokenizer")
                tokenizers[lang] = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=False
                )
                logger.info(f"✓ Loaded slow tokenizer for {lang}")
            
            # Load model
            models[lang] = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                num_labels=2
            )
            models[lang].eval()
            loaded_count += 1
            
            logger.info(f"✓ Successfully loaded {lang} model ({loaded_count}/{len(languages)})")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {lang} model: {str(e)}")
            logger.error(f"  Continuing with other models...")
            continue
    
    if loaded_count == 0:
        raise RuntimeError("No models could be loaded. Please check model files.")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Model loading complete: {loaded_count}/{len(languages)} models loaded")
    logger.info(f"  Available languages: {', '.join(models.keys())}")
    logger.info(f"{'='*60}\n")

@app.on_event("startup")
async def startup_event():
    """
    Load models and connect to database when the application starts
    """
    logger.info("\n" + "="*60)
    logger.info("Starting Multilingual Metaphor Detection API")
    logger.info("="*60 + "\n")
    
    try:
        load_models()
        logger.info("✓ Models loaded successfully\n")
    except Exception as e:
        logger.error(f"✗ Model loading failed: {str(e)}")
        logger.error("Please check that all model files are present in the models/ directory")
    
    # Load Whisper model - COMMENTED OUT: Using Web Speech API instead
    # global whisper_model
    # try:
    #     whisper_path = MODEL_BASE_PATH / "whisper"
    #     logger.info(f"Loading Whisper model from {whisper_path}...")
    #     whisper_model = whisper.load_model("base", download_root=str(whisper_path))
    #     logger.info("✓ Whisper model loaded successfully\n")
    # except Exception as e:
    #     logger.error(f"✗ Whisper model loading failed: {str(e)}")
    #     logger.warning("Speech-to-text feature will be disabled")
    
    # Connect to MongoDB
    try:
        await connect_to_mongodb()
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {str(e)}")
        logger.warning("History feature will be disabled")
    
    logger.info("✓ Application startup complete\n")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Close database connection on shutdown
    """
    logger.info("Shutting down application...")
    await close_mongodb_connection()
    logger.info("✓ Application shutdown complete")

@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed system information
    """
    import psutil
    import torch
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU info if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
            }
        else:
            gpu_info = {"gpu_available": False}
        
        return {
            "status": "System is running",
            "models_loaded": list(models.keys()),
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / 1024**3,
                "memory_total_gb": memory.total / 1024**3
            },
            "gpu_info": gpu_info,
            "gemini_api_configured": GEMINI_API_KEY is not None
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "System is running with limited monitoring",
            "models_loaded": list(models.keys()),
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Predict whether the input text contains metaphors
    Supports both single sentences and paragraphs with sentence-level analysis
    """
    try:
        text = input_data.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        # Check text length (increased for paragraph support)
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long. Please limit to 5000 characters.")
        
        # Check cache first
        cached_result = get_cached_prediction(text, input_data.interpretation_language)
        if cached_result:
            return PredictionResponse(**cached_result)
        
        # Detect language
        language = detect_language(text)
        logger.info(f"Detected language: {language}")
        
        # Check if model is loaded
        if language not in models:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=500,
                detail=f"Model for {language} is not available. Supported languages: {', '.join(available_models)}"
            )
        
        # Get model and tokenizer
        model = models[language]
        tokenizer = tokenizers[language]
        
        # Split text into sentences
        sentences = split_into_sentences(text)
        is_paragraph = len(sentences) > 1
        
        logger.info(f"Processing {'paragraph' if is_paragraph else 'single sentence'} with {len(sentences)} sentence(s)")
        
        # Analyze each sentence
        sentence_analyses = []
        metaphor_count = 0
        normal_count = 0
        total_confidence = 0.0
        anchor_text = ""  # Sticky anchor: persists across sentences until chain is broken
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # FIRST PASS: Standard Check - Analyze current sentence alone
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            sentence_label = "metaphor" if predicted_class == 1 else "normal"
            
            # If standard check returns Metaphor, update anchor for future sentences
            if sentence_label == "metaphor":
                anchor_text = sentence
                logger.info(f"Metaphor detected: Setting anchor_text = '{sentence[:50]}...'")
            
            # SECOND PASS: Context Check - If standard check returned Normal and anchor exists
            elif sentence_label == "normal" and anchor_text:
                # Concatenate anchor with current sentence for context-aware prediction
                combined_text = f"{anchor_text} {sentence}"
                logger.info(f"Standard check returned Normal, trying context check with anchor")
                
                inputs_context = tokenizer(
                    combined_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs_context = model(**inputs_context)
                    logits_context = outputs_context.logits
                    probabilities_context = torch.softmax(logits_context, dim=1)
                    predicted_class_context = torch.argmax(probabilities_context, dim=1).item()
                    confidence_context = probabilities_context[0][predicted_class_context].item()
                
                # If context check returns Metaphor, update label and confidence
                if predicted_class_context == 1:
                    sentence_label = "metaphor"
                    confidence = confidence_context
                    logger.info(f"Context check returned Metaphor! Updating label.")
                else:
                    # Both checks returned Normal, clear the anchor (chain is broken)
                    anchor_text = ""
                    logger.info(f"Both checks returned Normal, clearing anchor_text")
            
            # If standard check returned Normal and no anchor, chain is already broken
            elif sentence_label == "normal" and not anchor_text:
                logger.info(f"Standard check returned Normal, no anchor available")
            
            # Count labels for dominant label calculation
            if sentence_label == "metaphor":
                metaphor_count += 1
            else:
                normal_count += 1
            
            total_confidence += confidence
            
            logger.info(f"Sentence: '{sentence[:50]}...' -> {sentence_label} ({confidence:.4f})")
            
            # Generate Gemini interpretation for this sentence (run in thread pool)
            interpretations = await asyncio.to_thread(
                generate_gemini_interpretation,
                sentence,
                language,
                input_data.interpretation_language
            )
            
            # Get Gemini's prediction for verification
            gemini_prediction = get_gemini_prediction(sentence, language)
            
            # Verify the model's prediction with Gemini
            is_verified = False
            verification_status = ""
            
            if gemini_prediction["label"] != "error":
                is_verified = (gemini_prediction["label"] == sentence_label)
                if is_verified:
                    verification_status = "Verified by Gemini"
                else:
                    verification_status = f"Prediction mismatch with Gemini (Gemini predicted: {gemini_prediction['label']})"
            else:
                verification_status = "Verification failed: " + gemini_prediction.get("error", "Unknown error")
            
            # Create sentence analysis with verification info
            sentence_analysis = SentenceAnalysis(
                sentence=sentence,
                label=sentence_label,
                confidence=round(confidence, 4),
                interpretations=interpretations,
                is_verified=is_verified,
                verification_status=verification_status
            )
            
            sentence_analyses.append(sentence_analysis)
        
        # Determine overall label based on dominant frequency
        overall_label = "metaphor" if metaphor_count > normal_count else "normal"
        overall_confidence = total_confidence / len(sentences) if sentences else 0.0
        
        # Check overall verification status
        verified_sentences = [s for s in sentence_analyses if s.is_verified is True]
        is_fully_verified = len(verified_sentences) == len(sentence_analyses)
        verification_status = "Fully verified by Gemini" if is_fully_verified else "Partial or no verification"
        
        # Prepare response with proper serialization
        sentences_data = []
        for s in sentence_analyses:
            sentence_dict = {
                "sentence": s.sentence,
                "label": s.label,
                "confidence": s.confidence,
                "interpretations": {
                    "translation": s.interpretations.translation,
                    "literal": s.interpretations.literal,
                    "emotional": s.interpretations.emotional,
                    "philosophical": s.interpretations.philosophical,
                    "cultural": s.interpretations.cultural
                },
                "is_verified": s.is_verified,
                "verification_status": s.verification_status
            }
            sentences_data.append(sentence_dict)
            logger.info(f"Sentence interpretations: {sentence_dict['interpretations']}")
        
        response_data = {
            "language": language,
            "label": overall_label,
            "confidence": round(overall_confidence, 4),
            "text": text,
            "is_paragraph": is_paragraph,
            "sentences": sentences_data,
            "is_verified": is_fully_verified,
            "verification_status": verification_status,
            "interpretation_language": input_data.interpretation_language,
            # Legacy fields for backward compatibility
            "translation": sentence_analyses[0].interpretations.translation if sentence_analyses else "",
            "explanation": ""  # This will be populated by the frontend if needed
        }
        
        # Cache the result
        cache_prediction(text, response_data, input_data.interpretation_language)
        
        # Save to database (async, don't wait for it)
        try:
            await save_prediction(response_data.copy())
        except Exception as db_error:
            logger.warning(f"Failed to save to database: {str(db_error)}")
            # Don't fail the request if database save fails
        
        return PredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text from source language to English
    Uses googletrans library for quick translation (free, no API key needed)
    For production, consider IndicTrans2 or Google Cloud Translation API
    """
    try:
        logger.info(f"Translation request for {request.source_language}: {request.text}")
        
        # Try to use googletrans if available
        try:
            from googletrans import Translator
            translator = Translator()
            
            # Map language codes
            lang_map = {
                'hindi': 'hi',
                'tamil': 'ta',
                'kannada': 'kn'
            }
            
            source_lang = lang_map.get(request.source_language, 'auto')
            result = translator.translate(request.text, src=source_lang, dest='en')
            translated_text = result.text
            
            logger.info(f"Translation successful: {translated_text}")
            
        except ImportError:
            # Fallback if googletrans not installed
            logger.warning("googletrans not installed, using placeholder")
            translated_text = f"[Translation of '{request.text}' from {request.source_language} to English. Install 'googletrans==4.0.0-rc1' for automatic translation.]"
        except Exception as trans_error:
            # Fallback if translation fails
            logger.error(f"Translation error: {str(trans_error)}")
            translated_text = f"[Translation temporarily unavailable. Original text: '{request.text}']"
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# COMMENTED OUT: Using Web Speech API instead
# @app.post("/speech", response_model=SpeechResponse)
# async def speech_to_text(file: UploadFile = File(...), language: str = None):
#     """
#     Convert speech to text using Whisper
#     Supports Hindi, Tamil, Telugu, and Kannada
#     
#     Args:
#         file: Audio file to transcribe
#         language: Target language (hindi, tamil, telugu, kannada). If not provided, auto-detects.
#     """
#     try:
#         # Check if Whisper model is loaded
#         if whisper_model is None:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Speech recognition service not available. Whisper model not loaded."
#             )
#        
#        # Check file size (limit to 10MB)
#        file_content = await file.read()
#        if len(file_content) > 10 * 1024 * 1024:
#            raise HTTPException(
#                status_code=413,
#                detail="Audio file too large. Maximum size is 10MB."
#            )
#        
#        logger.info(f"Received audio file: {file.filename}, size: {len(file_content)} bytes")
#        if language:
#            logger.info(f"Requested language: {language}")
#        
#        # Save uploaded file to temporary location
#        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
#            temp_audio.write(file_content)
#            temp_audio_path = temp_audio.name
#        
#        try:
#            # Map language names to Whisper language codes
#            lang_to_whisper = {
#                'hindi': 'hi',
#                'tamil': 'ta',
#                'telugu': 'te',
#                'kannada': 'kn'
#            }
#            
#            # Determine which language to use for transcription
#            whisper_lang = lang_to_whisper.get(language.lower()) if language else None
#            
#            # Transcribe audio using Whisper
#            logger.info(f"Transcribing audio file: {temp_audio_path}")
#            if whisper_lang:
#                logger.info(f"Using specified language: {language} (Whisper code: {whisper_lang})")
#            else:
#                logger.info(f"Auto-detecting language")
#            
#            result = whisper_model.transcribe(
#                temp_audio_path,
#                language=whisper_lang,  # Use specified language or None for auto-detect
#                task="transcribe"
#            )
#            
#            transcribed_text = result["text"].strip()
#            detected_language = result.get("language", "unknown")
#            
#            logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")
#            logger.info(f"Detected language: {detected_language}")
#            
#            # Map Whisper language codes to our supported languages
#            whisper_lang_map = {
#                'hi': 'hindi',
#                'ta': 'tamil',
#                'te': 'telugu',
#                'kn': 'kannada'
#            }
#            
#            mapped_language = whisper_lang_map.get(detected_language, detected_language)
#            
#            # Check if detected language is supported
#            if detected_language not in whisper_lang_map:
#                logger.warning(f"Detected language '{detected_language}' is not in supported list")
#                return SpeechResponse(
#                    text=transcribed_text,
#                    success=True,
#                    message=f"Transcription successful, but detected language '{detected_language}' may not be supported for metaphor detection.",
#                    language=mapped_language
#                )
#            
#            return SpeechResponse(
#                text=transcribed_text,
#                success=True,
#                message="Transcription successful",
#                language=mapped_language
#            )
#            
#        finally:
#            # Clean up temporary file
#            try:
#                os.unlink(temp_audio_path)
#            except Exception as cleanup_error:
#                logger.warning(f"Failed to delete temporary file: {cleanup_error}")
#        
#    except HTTPException:
#        raise
#    except Exception as e:
#        logger.error(f"Speech-to-text error: {str(e)}")
#        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """
    Get detailed information about loaded models
    """
    try:
        model_info = {}
        
        for lang, model in models.items():
            try:
                # Get model parameters count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Get model config if available
                config_info = {}
                if hasattr(model, 'config'):
                    config_info = {
                        "model_type": getattr(model.config, 'model_type', 'unknown'),
                        "hidden_size": getattr(model.config, 'hidden_size', 'unknown'),
                        "num_layers": getattr(model.config, 'num_hidden_layers', 'unknown'),
                        "vocab_size": getattr(model.config, 'vocab_size', 'unknown')
                    }
                
                model_info[lang] = {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params * 4 / 1024 / 1024,  # Approximate size in MB
                    "config": config_info,
                    "tokenizer_vocab_size": len(tokenizers[lang]) if lang in tokenizers else 0
                }
            except Exception as e:
                model_info[lang] = {"error": str(e)}
        
        return {
            "models": model_info,
            "total_models": len(models),
            "supported_languages": list(LANGUAGE_MAP.values())
        }
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# ==================== HISTORY ENDPOINTS ====================

@app.get("/history")
async def get_history(
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    language: Optional[str] = Query(None, description="Filter by language"),
    label: Optional[str] = Query(None, description="Filter by label (metaphor/normal)")
):
    """
    Get prediction history with optional filters
    """
    try:
        history = await get_prediction_history(limit=limit, skip=skip, language=language, label=label)
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Failed to get history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/history/{prediction_id}")
async def get_history_item(prediction_id: str):
    """
    Get a specific prediction by ID
    """
    try:
        prediction = await get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")


@app.delete("/history/{prediction_id}")
async def delete_history_item(prediction_id: str):
    """
    Delete a specific prediction from history
    """
    try:
        success = await delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found or already deleted")
        return {"success": True, "message": "Prediction deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")


@app.delete("/history")
async def clear_history():
    """
    Clear all prediction history
    """
    try:
        count = await clear_all_history()
        return {
            "success": True,
            "message": f"Cleared {count} predictions from history",
            "deleted_count": count
        }
    except Exception as e:
        logger.error(f"Failed to clear history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@app.get("/statistics")
async def get_stats():
    """
    Get statistics about predictions
    """
    try:
        stats = await get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
