import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained 2D CNN model
@st.cache_resource
def load_emotion_model():
    return load_model("models.h5")

# Emotion class labels and their corresponding colors
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 255, 0), # Cyan
    'Neutral': (255, 255, 255) # White
}

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.model = load_emotion_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_history = deque(maxlen=10)
        self.current_emotion = "Neutral"
        self.fps = 0
        self.prev_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract and preprocess the face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

            # Predict emotion
            prediction = self.model.predict(roi_reshaped, verbose=0)
            emotion_idx = np.argmax(prediction)
            label = class_labels[emotion_idx]
            confidence = prediction[0][emotion_idx] * 100

            # Update emotion history
            self.emotion_history.append(label)
            self.current_emotion = max(set(self.emotion_history), key=self.emotion_history.count)

            # Get color for current emotion
            color = emotion_colors.get(self.current_emotion, (255, 255, 255))

            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{self.current_emotion}: {confidence:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display FPS
        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display emotion history
        history_text = "Recent Emotions: " + ", ".join(self.emotion_history)
        cv2.putText(img, history_text, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

def main():
    st.title("Real-Time Emotion Detection")
    st.write("This app detects emotions in real-time using your webcam.")

    # Add some information about the app
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses a deep learning model to detect emotions in real-time.
        The model classifies emotions into 7 categories:
        - Angry
        - Disgust
        - Fear
        - Happy
        - Sad
        - Surprise
        - Neutral
        """
    )

    # Add a checkbox to show/hide the webcam feed
    show_webcam = st.checkbox("Show Webcam Feed", value=True)

    if show_webcam:
        webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=EmotionDetector,
            async_transform=True,
        )

    # Add a section for emotion statistics
    st.subheader("Emotion Statistics")
    st.write("The current emotion detection is shown in the webcam feed.")
    st.write("Recent emotions are displayed at the bottom of the video.")

if __name__ == "__main__":
    main() 