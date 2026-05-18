"""
Emotion Detection module using IBM Watson NLP library.
Calls the Watson NLP Emotion Predict endpoint and returns
structured emotion analysis results.
"""

import json
import requests


def emotion_detector(text_to_analyze):
    """
    Detect emotions in the provided text using Watson NLP.

    Args:
        text_to_analyze (str): The input text to analyse for emotions.

    Returns:
        dict: A dictionary containing scores for anger, disgust, fear,
              joy, and sadness, plus the dominant emotion label.
              Returns None values when input is blank or invalid.
    """
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
    except requests.exceptions.RequestException:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    # Handle non-200 responses (e.g. 400 Bad Request for blank input)
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    if response.status_code != 200:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    # Task 3: Format the output
    response_data = json.loads(response.text)
    emotions = response_data["emotionPredictions"][0]["emotion"]

    anger_score = emotions.get("anger", 0)
    disgust_score = emotions.get("disgust", 0)
    fear_score = emotions.get("fear", 0)
    joy_score = emotions.get("joy", 0)
    sadness_score = emotions.get("sadness", 0)

    emotion_scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {**emotion_scores, "dominant_emotion": dominant_emotion}
