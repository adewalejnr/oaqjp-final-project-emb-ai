"""
Unit tests for the EmotionDetection module.
Tests cover: positive emotions, negative emotions, blank input error handling.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EmotionDetection import emotion_detector


def _mock_response(emotions_dict, status_code=200):
    """Helper: build a mock requests.Response for given emotions."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    payload = {
        "emotionPredictions": [
            {"emotion": emotions_dict}
        ]
    }
    mock_resp.text = json.dumps(payload)
    return mock_resp


class TestEmotionDetector(unittest.TestCase):
    """Test suite for emotion_detector function."""

    # ------------------------------------------------------------------ #
    # Happy-path tests                                                     #
    # ------------------------------------------------------------------ #

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_is_dominant_for_happy_text(self, mock_post):
        """Joy should be the dominant emotion for clearly positive text."""
        mock_post.return_value = _mock_response(
            {"anger": 0.02, "disgust": 0.01, "fear": 0.03, "joy": 0.92, "sadness": 0.02}
        )
        result = emotion_detector("I am thrilled about this, it is a great victory for us!")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_is_dominant_for_angry_text(self, mock_post):
        """Anger should be the dominant emotion for aggressive text."""
        mock_post.return_value = _mock_response(
            {"anger": 0.85, "disgust": 0.04, "fear": 0.03, "joy": 0.02, "sadness": 0.06}
        )
        result = emotion_detector("I am really mad about this!")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_is_dominant_for_disgusting_text(self, mock_post):
        """Disgust should be the dominant emotion for repulsive text."""
        mock_post.return_value = _mock_response(
            {"anger": 0.05, "disgust": 0.87, "fear": 0.02, "joy": 0.01, "sadness": 0.05}
        )
        result = emotion_detector("I feel disgusted just thinking about it.")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_is_dominant_for_sad_text(self, mock_post):
        """Sadness should be the dominant emotion for sorrowful text."""
        mock_post.return_value = _mock_response(
            {"anger": 0.03, "disgust": 0.02, "fear": 0.04, "joy": 0.01, "sadness": 0.90}
        )
        result = emotion_detector("It is really sad how things have turned out")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_is_dominant_for_fearful_text(self, mock_post):
        """Fear should be the dominant emotion for alarming text."""
        mock_post.return_value = _mock_response(
            {"anger": 0.04, "disgust": 0.02, "fear": 0.88, "joy": 0.01, "sadness": 0.05}
        )
        result = emotion_detector("I am scared to death by this news!")
        self.assertEqual(result["dominant_emotion"], "fear")

    # ------------------------------------------------------------------ #
    # Error-handling tests                                                 #
    # ------------------------------------------------------------------ #

    def test_blank_input_returns_none_dominant(self):
        """Blank or whitespace input must return dominant_emotion as None."""
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])

    def test_blank_input_returns_none_scores(self):
        """Blank input must return None for all individual emotion scores."""
        result = emotion_detector("   ")
        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIsNone(result[key])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_400_response_returns_none_dominant(self, mock_post):
        """A 400 response from Watson must return dominant_emotion as None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_post.return_value = mock_resp
        result = emotion_detector("xyz")
        self.assertIsNone(result["dominant_emotion"])

    # ------------------------------------------------------------------ #
    # Output-structure tests                                               #
    # ------------------------------------------------------------------ #

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_result_contains_all_keys(self, mock_post):
        """Result dict must contain all five emotion keys plus dominant_emotion."""
        mock_post.return_value = _mock_response(
            {"anger": 0.1, "disgust": 0.1, "fear": 0.1, "joy": 0.6, "sadness": 0.1}
        )
        result = emotion_detector("Some text")
        expected_keys = {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"}
        self.assertEqual(set(result.keys()), expected_keys)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_scores_are_floats(self, mock_post):
        """All numeric emotion scores must be floats (not strings)."""
        mock_post.return_value = _mock_response(
            {"anger": 0.05, "disgust": 0.05, "fear": 0.05, "joy": 0.80, "sadness": 0.05}
        )
        result = emotion_detector("Great day!")
        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIsInstance(result[key], float)


if __name__ == "__main__":
    unittest.main()
