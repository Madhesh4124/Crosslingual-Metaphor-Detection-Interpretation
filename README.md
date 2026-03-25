# Crosslingual Metaphor Detection System

A production-ready NLP system for detecting metaphors in **Hindi, Tamil, Telugu, and Kannada** using fine-tuned transformer models.
The project includes a **FastAPI backend**, **React frontend**, training scripts, and detailed documentation.
All trained models are hosted on **Hugging Face** and are automatically downloaded at runtime.

---

## 🚀 Features

* Crosslingual metaphor detection (4 Indian languages)
* Sentence-level and paragraph-level analysis
* **Cross-lingual AI interpretation** (Get explanations in English, Hindi, Tamil, Telugu, or Kannada)
* 5-layer AI interpretation (translation, literal, emotional, philosophical, cultural)
* REST API built with FastAPI
* Interactive React frontend
* Speech-to-text input (Web Speech API)
* Optional history tracking using MongoDB
* Clean separation of code and models (industry best practice)

---

## 📝 How Paragraph Detection & Interpretation Works

When a paragraph (multi-sentence input) is submitted:

* The backend splits the text into individual sentences using both English and Indic punctuation.
* Each sentence is analyzed for metaphoric content **individually**.
* For every sentence, a 5-layer AI interpretation is generated in the **selected target language** (English, Hindi, Telugu, Tamil, or Kannada).

### Context-Aware Metaphor Detection

* If a sentence is detected as a metaphor, it becomes the *anchor* for the next sentence.
* If the next sentence is classified as *normal* but follows a metaphor, the backend re-evaluates it using the previous metaphor sentence as context.
* If the context-aware check finds a metaphor, the label is updated to **metaphor**.
* If two consecutive sentences are *normal*, the context chain is broken.
* This allows metaphors that depend on prior context to be detected, not just isolated sentences.

The API returns a detailed, per-sentence analysis and interpretation for the whole paragraph, along with a summary label and confidence.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **FastAPI**, Uvicorn
* **Transformers (Hugging Face)**
* **PyTorch**
* **MongoDB** (async, Motor)
* **React (Vite)**
* **Google Gemini API** (for interpretations)

---

## 📂 Folder Structure

```
.github/           # GitHub workflows & templates
.vscode/           # VSCode settings
backend/           # FastAPI backend, API, DB logic
frontend/          # React frontend
datasets/          # Example / test datasets
training_code/     # Model training scripts
models/            # (Ignored) Local models – hosted on Hugging Face
.env               # Local secrets (ignored)
.env.example       # Example environment variables
requirements.txt   # Python dependencies
README.md          # Project documentation
LICENSE            # License file
MONGODB_SETUP.md   # MongoDB setup guide
```

---

## 🤗 Hugging Face Models

The trained models are hosted on Hugging Face and are **automatically downloaded** using `from_pretrained()`.

* **Hindi (XLM-R)**
  [https://huggingface.co/Madhesh4124/hindi-metaphor-xlm](https://huggingface.co/Madhesh4124/hindi-metaphor-xlm)

* **Tamil (XLM-R)**
  [https://huggingface.co/Madhesh4124/tamil-metaphor-xlm](https://huggingface.co/Madhesh4124/tamil-metaphor-xlm)

* **Telugu (MuRIL)**
  [https://huggingface.co/Madhesh4124/telugu-metaphor-muril](https://huggingface.co/Madhesh4124/telugu-metaphor-muril)

* **Kannada (BERT)**
  [https://huggingface.co/Madhesh4124/kannada-metaphor-bert](https://huggingface.co/Madhesh4124/kannada-metaphor-bert)

> ⚠️ The `models/` directory is intentionally **not included** in this repository.

---

## ⚙️ Setup Instructions

### 1️⃣ Create Environment File

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for history feature)
# Use localhost OR placeholders for MongoDB Atlas
MONGODB_URL=mongodb://localhost:27017
# OR
# MONGODB_URL=mongodb+srv://<username>:<password>@<cluster-url>/<db-name>

MONGODB_DB_NAME=metaphor_detector
```

Get a Gemini API key from:
[https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ (Optional) Set Up MongoDB

MongoDB is **optional**.
The application works without it, but history will be disabled.

See: [MONGODB_SETUP.md](MONGODB_SETUP.md)

---

## ▶️ Running the Application

### Start Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend will be available at:

```
http://localhost:8000
```

---

### Start Frontend (New Terminal)

```bash
cd frontend
npm install
npm run dev
```

Frontend will open at:

```
http://localhost:3000
```

---

## 🧪 Quick Test

1. Copy this Hindi metaphor:
   `वह आग में घी डाल रहा है`
2. Paste it into the input box
3. Click **Analyze**
4. View the multi-layer interpretation 🎉

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

---

### Predict Metaphor

```http
POST /predict
Content-Type: application/json
```

```json
{
  "text": "वह आसमान छू रहा है",
  "is_paragraph": false,
  "interpretation_language": "english" 
}
```
*`interpretation_language` options: `english`, `hindi`, `tamil`, `telugu`, `kannada`*
```

---

### Other Endpoints

```http
GET    /history
GET    /statistics
DELETE /history/{prediction_id}
DELETE /history
```

---

## ⚙️ Configuration

| Variable          | Required | Description                                       |
| ----------------- | -------- | ------------------------------------------------- |
| `GEMINI_API_KEY`  | Yes      | Gemini API key for interpretations                |
| `MONGODB_URL`     | No       | MongoDB connection string (use placeholders only) |
| `MONGODB_DB_NAME` | No       | Database name                                     |

---

## 🎓 For Students & Developers

This project demonstrates:

* Crosslingual NLP using Transformers
* FastAPI-based REST APIs
* Full-stack integration (React + ML backend)
* Clean ML project structuring
* Proper model hosting with Hugging Face

---

## 📄 License

This project is released for **educational and academic purposes**.

---

## 🙏 Acknowledgments

* Hugging Face (Transformers & Model Hub)
* Google Gemini API
* FastAPI community
* React community

---

**Happy Coding 🚀**
If you find this project useful, feel free to ⭐ the repository!


