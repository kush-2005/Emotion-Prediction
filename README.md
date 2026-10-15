# Emotion Prediction

A web app that classifies the emotion expressed in a sentence. It uses a Bidirectional GRU (BiGRU) text classifier built with TensorFlow/Keras and served through a FastAPI backend with a simple HTML/JS front end.

**Emotions:** sadness 😢 · joy 😄 · love ❤️ · anger 😠 · fear 😨 · surprise 😲

## Features

- BiGRU model trained on labelled emotion text (see `emotion_prediction.ipynb`)
- REST API with input validation (Pydantic)
- Returns the top emotion, its confidence, and the full probability breakdown
- Built-in web UI served at `/`
- Health-check endpoint

## Project structure

```
.
├── Artifacts/
│   ├── BiGRU_Model.keras    # trained model
│   └── tokenizer.pkl        # fitted Keras tokenizer
├── static/                  # front end (index.html, script.js, style.css)
├── emotion_prediction.ipynb # data exploration and model training
├── main.py                  # FastAPI application
├── requirements.txt
└── runtime.txt              # Python version (3.11.9)
```

## Getting started

Requires Python 3.11.

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs for the interactive API docs.

## API

### `GET /health`

```json
{ "status": "Server is running", "model_loaded": true }
```

### `POST /predict`

Request body (`text` is 1–2000 characters):

```json
{ "text": "I feel so happy and excited" }
```

Response:

```json
{
  "text": "I feel so happy and excited",
  "predicted_emotion": "joy",
  "confidence": 0.97,
  "all_probabilites": {
    "sadness": 0.01,
    "joy": 0.97,
    "love": 0.01,
    "anger": 0.00,
    "fear": 0.00,
    "surprise": 0.01
  }
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy and excited"}'
```

The response field is spelled `all_probabilites` in the current API.

## How it works

1. The input is lowercased, apostrophes are removed, punctuation is stripped and whitespace is normalised.
2. The tokenizer converts words to integer IDs.
3. Sequences are padded or truncated to 50 tokens.
4. The BiGRU model outputs a probability for each of the six emotions.
