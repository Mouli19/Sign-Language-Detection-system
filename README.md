# Sign-Language-Detection-system

An original webcam-based sign language detection system with a local Flask
backend and a browser website.

The backend uses MediaPipe hand landmarks and an original rule-based classifier
for common static hand signs. It is designed as a working starter system that
you can run locally, study, and extend with trained machine-learning models.

## Features

- Live webcam website.
- Flask prediction API.
- MediaPipe hand landmark detection.
- Original static-sign classifier.
- Prediction confidence and hand skeleton overlay.
- Clean, responsive web interface.

## Detected Signs

The starter classifier recognizes these static signs:

- Open Palm / Stop
- Fist / Yes
- Peace
- Point One
- I Love You
- Thumbs Up
- OK
- Call Me

Lighting, camera angle, and hand distance affect accuracy. Keep your hand clear,
upright, and inside the camera frame.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── sign_detector.py
├── static
│   ├── app.js
│   └── styles.css
└── templates
    └── index.html
```

## How It Works

1. The website opens your webcam with `getUserMedia`.
2. Every few frames, the browser sends a small image to `/api/predict`.
3. The backend finds hand landmarks with MediaPipe.
4. Landmark geometry is converted into finger states and shape measurements.
5. The classifier returns a sign label, confidence score, and landmark points.

## Extend It

For a larger project, collect landmark samples and replace the rule-based logic
in `sign_detector.py` with a trained classifier such as Random Forest, SVM, LSTM,
or a small neural network.
app.py
