# Emotion-Based Music Recommendation System

**Built by Ridaa**

A real-time web application that detects facial expressions through a live webcam feed using a custom-trained Convolutional Neural Network (CNN) and recommends mood-matched Spotify tracks.

## Features
- **Real-Time Face Detection:** Uses OpenCV's Haar Cascade Classifier to detect faces in live webcam frames.
- **Facial Emotion Recognition:** Classifies 7 facial expressions (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`) using a TensorFlow/Keras CNN trained on the FER-2013 dataset.
- **Temporal Prediction Smoothing:** Uses a 15-frame rolling majority-vote buffer (`collections.deque`) and class-weight calibration for stable, flicker-free predictions.
- **Dynamic Music Recommendations:** Maps detected emotions to music moods (`Happy`, `Sad`, `Energetic`, `Calm`) and serves live song playlists via a Flask REST API.

## Tech Stack
- **Deep Learning & Computer Vision:** Python 3.10, TensorFlow / Keras, OpenCV (`cv2`), NumPy
- **Backend & Data Processing:** Flask, Pandas
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## Project Structure
```text
emotion-music-app/
├── app.py                                       # Flask web server & real-time inference
├── requirements.txt                             # Python dependencies
├── README.md                                    # Project documentation
├── dataset/
│   └── data_moods.csv                           # Spotify mood-labeled music dataset
├── model/
│   └── emotion_model.h5                         # Trained CNN model weights
├── notebook/
│   └── emotion-based-music-recommendation.ipynb # Model training & evaluation notebook
└── templates/
    └── index.html                               # Web interface
```

## How to Run Locally
1. Clone this repository and navigate into the project folder:
   ```bash
   git clone https://github.com<your-username>/emotion-music-app.git
   cd emotion-music-app
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the **FER-2013** dataset (`train` and `test` folders) from Kaggle into `dataset/` if you wish to retrain the model, or run the pre-trained model directly:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.