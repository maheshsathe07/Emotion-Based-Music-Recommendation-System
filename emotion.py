import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
model_path = "E:\\College\\TY\\ML\\Project\\Spotify-Song-Recommendation-Engine-master\\facialemotionmodel.h5"
emotion_model = load_model(model_path)

# Emotion classes
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(face, model):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    prediction = model.predict(face)
    return emotions[np.argmax(prediction)]

def main():
    st.title("Emotion Detection")

    start_button = st.button('Start Camera')
    placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        detected_emotion = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                detected_emotion = detect_emotion(face, emotion_model)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Update streamlit image
            placeholder.image(frame, channels="BGR", caption="Detecting Emotion")


            # Break loop if emotion is detected
            if detected_emotion:
                break

        cap.release()
        st.write(f"Detected Emotion: **{detected_emotion}**")

if __name__ == "__main__":
    main()
