# Emotion Detector

AI-powered emotion analysis web application built on IBM Watson NLP.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python server.py
# → http://localhost:5000
```

## Tasks Completed

| # | Task | Status |
|---|------|--------|
| 1 | Clone / project setup | ✅ |
| 2 | Emotion detection via Watson NLP | ✅ |
| 3 | Formatted output (anger/disgust/fear/joy/sadness + dominant) | ✅ |
| 4 | Packaged as `EmotionDetection` installable module | ✅ |
| 5 | 10 unit tests — all passing | ✅ |
| 6 | Flask web deployment (`/` and `/emotionDetector`) | ✅ |
| 7 | Error handling (blank input → 400 + message) | ✅ |
| 8 | pylint static analysis — **10.00 / 10** | ✅ |

## API

```
GET /emotionDetector?textToAnalyze=<your+text>
```

**200 response:**
```json
{
  "anger": 0.012,
  "disgust": 0.005,
  "fear": 0.018,
  "joy": 0.923,
  "sadness": 0.042,
  "dominant_emotion": "joy",
  "formatted_output": "For the given statement..."
}
```

**400 response (blank input):**
```json
{ "error": "Invalid text! Please try again." }
```

## Project Structure

```
emotion_detector/
├── EmotionDetection/
│   ├── __init__.py
│   └── emotion_detection.py   # Core Watson NLP logic
├── templates/
│   └── index.html             # Frontend UI
├── tests/
│   └── test_emotion_detection.py
├── server.py                  # Flask app
├── setup.py
└── requirements.txt
```
