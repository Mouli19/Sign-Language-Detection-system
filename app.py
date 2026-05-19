from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from sign_detector import SignLanguageDetector


app = Flask(__name__)
CORS(app)
detector = SignLanguageDetector()


def decode_frame(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode camera frame.")
    return frame


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        return jsonify({"error": "Missing image field."}), 400

    try:
        frame = decode_frame(image)
        result = detector.predict(frame)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.get("/api/signs")
def signs():
    return jsonify({"signs": detector.supported_signs})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
