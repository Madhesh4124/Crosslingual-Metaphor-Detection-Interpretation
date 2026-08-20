---
title: Metaphor Detection Backend
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Crosslingual Metaphor Detection & Interpretation System

A production-grade NLP architecture for detecting and explaining metaphors across **Hindi, Tamil, Telugu, and Kannada** using fine-tuned transformer models, explainable AI (XAI) feature attributions, and 5-layer LLM cognitive interpretations.

🌐 **Live Demo**: [https://crosslingual-metaphor-detection-interpretation-kovgohbol.vercel.app/](https://crosslingual-metaphor-detection-interpretation-kovgohbol.vercel.app/)

---

## 🏛️ System Architecture Overview

The system combines dedicated fine-tuned transformer encoders with gradient saliency attribution and generative LLM interpretation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REACT FRONTEND (Vite)                            │
│  - On-screen script keyboards (हिंदी / தமிழ் / తెలుగు / ಕನ್ನಡ)              │
│  - Web Speech API real-time microphone input                                │
│  - Multi-lingual target output selector (EN / HI / TA / TE / KN)            │
│  - Interactive XAI Saliency Badges & 5-Layer Cognitive Interpretations      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP POST /predict
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND ORCHESTRATION                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Script & Language Auto-Detector (Unicode Range + LangDetect)       │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ 2. Indic Sentence Tokenizer & Punctuation Segmenter (। . ? !)         │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ 3. On-Demand Lazy Model Loader (Per-Language PyTorch Checkpoints)     │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│         ┌────────────────────────────┴────────────────────────────┐         │
│         ▼                                                         ▼         │
│  ┌──────────────────────────────┐       ┌──────────────────────────────┐    │
│  │ 4. Two-Pass Context Engine   │       │ 5. PyTorch XAI Gradient      │    │
│  │    & Temperature Scaler      │       │    Embedding Saliency Norm   │    │
│  └──────────────┬───────────────┘       └──────────────┬───────────────┘    │
│                 │                                      │                    │
│                 └────────────────────┬─────────────────┘                    │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 6. Gemma 4 26B (gemma-4-26b-a4b-it via Google GenAI)                  │  │
│  │    - 5-Layer Semantic & Cultural Interpretation (Target Language)    │  │
│  │    - Secondary Classification Verification                            │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ 7. Async MongoDB Atlas Persistence (Predictions & History Stats)      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Algorithms & Computational Pipeline

### 1. Language & Script Detection
Input text is parsed through a hybrid detection hierarchy:
1. **Direct Unicode Range Mapping** (Highest reliability for Indic scripts):
   - Devanagari (Hindi): `\u0900` to `\u097F`
   - Tamil: `\u0B80` to `\u0BFF`
   - Telugu: `\u0C00` to `\u0C7F`
   - Kannada: `\u0C80` to `\u0CFF`
2. **LangDetect Engine**: Serves as a fallback for mixed or Romanized inputs.

---

### 2. Fine-Tuned Transformer Models
Each language runs on an optimized transformer architecture fine-tuned specifically for metaphor binary classification (`metaphor` vs. `normal`):

| Language | Base Architecture | Hugging Face Hub Model |
| :--- | :--- | :--- |
| **Hindi** | XLM-RoBERTa Base | [`Madhesh4124/hindi-metaphor-xlm`](https://huggingface.co/Madhesh4124/hindi-metaphor-xlm) |
| **Tamil** | XLM-RoBERTa Base | [`Madhesh4124/tamil-metaphor-xlm`](https://huggingface.co/Madhesh4124/tamil-metaphor-xlm) |
| **Telugu** | MuRIL (Multilingual Representations for Indic Languages) | [`Madhesh4124/telugu-metaphor-muril`](https://huggingface.co/Madhesh4124/telugu-metaphor-muril) |
| **Kannada** | Indic-BERT | [`Madhesh4124/kannada-metaphor-bert`](https://huggingface.co/Madhesh4124/kannada-metaphor-bert) |

---

### 3. Temperature Scaling & Calibration
Raw neural network logits often output overconfident probabilities for RoBERTa architectures and underconfident spreads for BERT/MuRIL on narrow margins. Calibrated confidence is computed via temperature-scaled softmax:

$$\hat{P}(y = k \mid x) = \frac{\exp(z_k / T)}{\sum_{j} \exp(z_j / T)}$$

- **Hindi / Tamil (XLM-RoBERTa)**: $T = 2.2$ (Smoothes extreme $>99.9\%$ overconfidence).
- **Telugu / Kannada (MuRIL / BERT)**: $T = 0.35$ (Amplifies narrow $52-60\%$ margins to true certainty).

---

### 4. Two-Pass Context-Aware Chaining
In multi-sentence paragraphs, metaphors frequently span across sentence boundaries where subsequent sentences appear literal in isolation:

```
Sentence 1: "उसका दिल पत्थर है।" (His heart is stone.) -> [Metaphor] -> Set Anchor = S1
Sentence 2: "कोई बात असर नहीं करती।" (Nothing affects him.) -> Pass 1: [Literal]
                                                            Pass 2: Context Evaluation [S1 + S2] -> [Metaphor]
```

- **Pass 1 (Isolated Evaluation)**: Evaluates $S_i$ independently.
  - If $S_i = \text{Metaphor}$, it updates the active anchor: $\text{Anchor} \leftarrow S_i$.
- **Pass 2 (Context Evaluation)**: If $S_i = \text{Literal}$ and an active anchor exists:
  - Concatenates: $S_{\text{context}} = [\text{Anchor}] \oplus [S_i]$.
  - If the concatenated forward pass classifies as $\text{Metaphor}$, the label for $S_i$ is updated to **Metaphor**.
  - If both passes return **Literal**, the context chain is broken and $\text{Anchor} \leftarrow \emptyset$.

---

### 5. Explainable AI (XAI) Embedding Gradient Saliency
To explain *why* the neural network made a classification decision, token-level feature attribution is calculated via backpropagation through the model's word embeddings layer:

1. Let $E \in \mathbb{R}^{L \times D}$ be the input embedding tensor for sequence tokens $t_1, t_2, \dots, t_L$.
2. Compute the gradient of the target class logit $z_{\text{target}}$ with respect to the input embeddings:

$$G_i = \nabla_{E_i} z_{\text{target}}$$

3. The saliency score for each token $i$ is calculated using the $L_2$ Euclidean norm of its gradient vector:

$$S_i = \|G_i\|_2$$

4. Saliency values are normalized into percentage weights across the sequence:

$$P_i = \frac{S_i}{\sum_{j=1}^{L} S_j} \times 100\%$$

5. **Dynamic Key Trigger Threshold**:
   A token is marked as a **Key Trigger** if:

$$P_i \ge (\mu_S + 0.5 \cdot \sigma_S) \quad \text{OR} \quad (\max(P) - P_i) \le 1.5\%$$

This dual criterion flags both statistically significant tokens above the standard deviation and tightly clustered co-triggers (e.g., `"उसका"` and `"चाँद"` in `"उसका चेहरा चाँद है"`).

---

### 6. 5-Layer Cognitive Interpretation (Gemma 4 26B)
For every detected metaphor, the backend queries **Gemma 4 26B** (`gemma-4-26b-a4b-it`) using structured few-shot prompting to output five perspectives in the selected language:

1. **Translation**: Direct idiomatic cross-lingual translation.
2. **Literal**: Word-for-word grammatical translation.
3. **Emotional**: Mood, affect, and emotional resonance conveyed.
4. **Philosophical**: Abstract life insight or metaphysical meaning.
5. **Cultural**: Indic cultural context, folklore, and metaphorical traditions.

---

## ⚡ Memory & Performance Optimization

* **On-Demand Lazy Loading**: PyTorch model weights are only loaded into RAM/VRAM when a request in that specific language is received, ensuring instant server startups and minimal idle memory footprints.
* **In-Memory Hash Caching**: MD5-hashed cache for identical predictions and LLM calls with a 1-hour TTL ($3600\text{s}$).
* **Fast-Fail Database Connectors**: MongoDB Atlas client initialized with `serverSelectionTimeoutMS=2000` to prevent request stalling when cloud network drops occur.

---

## 📂 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server, endpoints, XAI & inference engine
│   ├── database.py          # Motor async MongoDB connector & history operations
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── App.css          # Core design system & theme variables
│   │   ├── History.jsx      # Historical analytics & record viewer
│   │   ├── History.css      # History modal styling
│   │   ├── VirtualKeyboard.jsx # On-screen native script keyboards
│   │   └── VirtualKeyboard.css # Virtual keyboard styling
│   └── package.json         # Frontend dependencies & Vite scripts
├── models/                  # Local model cache (downloaded from Hugging Face)
├── datasets/                # Evaluation & benchmark datasets
├── training_code/           # Training scripts for XLM-R, MuRIL & BERT
├── app.py                   # Hugging Face Spaces deployment entrypoint
└── README.md                # System documentation & architectural reference
```

---

## 🚀 Getting Started

### 1. Environment Setup
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_genai_api_key_here
MONGODB_URL=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=metaphor_detector
```

### 2. Backend Execution
Activate your Python environment and run:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Frontend Execution
In a separate terminal:

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

Navigate to `http://localhost:5173`.

---

## 🔌 API Reference

### POST `/predict`
**Request**:
```json
{
  "text": "उसका चेहरा चाँद है",
  "interpretation_language": "english"
}
```

**Response**:
```json
{
  "language": "hindi",
  "label": "metaphor",
  "confidence": 0.9421,
  "text": "उसका चेहरा चाँद है",
  "is_paragraph": false,
  "sentences": [
    {
      "sentence": "उसका चेहरा चाँद है",
      "label": "metaphor",
      "confidence": 0.9421,
      "interpretations": {
        "translation": "Her face is radiant like the moon.",
        "literal": "Her face moon is.",
        "emotional": "Conveys deep romantic admiration and aesthetic wonder.",
        "philosophical": "Reflects how human beauty mirrors celestial luminosity.",
        "cultural": "In classical Indian poetics (Kavya), comparing a beloved's face to the moon is a standard archetype."
      },
      "word_attributions": [
        {"word": "उसका", "score": 15.47, "is_key_trigger": true},
        {"word": "चेहरा", "score": 13.65, "is_key_trigger": false},
        {"word": "चाँद", "score": 15.26, "is_key_trigger": true},
        {"word": "है", "score": 14.70, "is_key_trigger": false}
      ],
      "decision_reasoning": "The model identified 'उसका', 'चाँद' as key attribution trigger(s) driving the METAPHOR decision.",
      "is_verified": true,
      "verification_status": "Verified by Gemini"
    }
  ]
}
```

---

## 📄 License
Released for **educational, academic, and research purposes**.
