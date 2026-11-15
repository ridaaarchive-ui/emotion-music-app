import cv2
import numpy as np
import pandas as pd
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify
from keras.models import load_model

app = Flask(__name__)

# Load trained emotion model and OpenCV built-in face detector
model = load_model('model/emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FER-2013 alphabetical folder order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Map detected facial emotions to the 'mood' column in data_moods.csv
emotion_to_mood = {
    'Happy': 'Happy',
    'Sad': 'Sad',
    'Angry': 'Energetic',
    'Fear': 'Calm',
    'Disgust': 'Calm',
    'Surprise': 'Energetic',
    'Neutral': 'Calm'
}

songs_df = pd.read_csv('dataset/data_moods.csv')
current_emotion = 'Neutral'

def generate_frames():
    global current_emotion
    camera = cv2.VideoCapture(0)
    emotion_history = deque(maxlen=15)

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (29, 185, 84), 2)
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            roi_gray = gray[y + pad_y:y + h - pad_y, x + pad_x:x + w - pad_x]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi, verbose=0)[0]
            preds *= np.array([2.0, 2.8, 1.9, 1.0, 0.80, 1.30, 1.9])
            emotion_history.append(emotion_labels[np.argmax(preds)])
            current_emotion = Counter(emotion_history).most_common(1)[0][0]
            confidence = int(np.max(preds) / np.sum(preds) * 100)
            label_text = f"{current_emotion} ({confidence}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (29, 185, 84), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_songs')
def get_songs():
    target_mood = emotion_to_mood.get(current_emotion, 'Calm')
    filtered = songs_df[songs_df['mood'] == target_mood]
    sample_songs = filtered.sample(n=min(10, len(filtered)))[['name', 'artist', 'album']].to_dict(orient='records')
    return jsonify({'emotion': current_emotion, 'songs': sample_songs})

if __name__ == '__main__':
    app.run(debug=True)