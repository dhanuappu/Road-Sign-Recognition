import os
# --- STABILITY: CPU MODE ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, Response, jsonify
import gc 
import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras # Use the compatibility wrapper
from tensorflow.keras.models import load_model
import pyttsx3 # <--- OFFLINE VOICE (No Internet Needed)
from collections import deque, Counter
import threading
from languages import get_sign_text

application = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'models/road_sign_model_final.h5' 
CONFIDENCE_THRESHOLD = 0.85 # High confidence to reduce wrong guesses
BUFFER_SIZE = 5

# Load Model
try:
    model = keras.models.load_model('models/road_sign_model_final.h5')
    print("Model loaded successfully using tf_keras!")
except Exception as e:
    print(f"Error loading model: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- OFFLINE VOICE ENGINE SETUP ---
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech
voice_lock = threading.Lock() # Prevents voice from overlapping

prediction_buffer = deque(maxlen=BUFFER_SIZE)
current_language = 'en' 
last_announced = None

def speak_sign(text):
    """
    OFFLINE THREADED AUDIO: Works without Internet.
    """
    def _speak():
        with voice_lock: # Ensure only one voice speaks at a time
            try:
                # Re-initialize engine inside thread for safety
                local_engine = pyttsx3.init()
                local_engine.setProperty('rate', 150)
                local_engine.say(text)
                local_engine.runAndWait()
            except Exception as e:
                print(f"Voice Error: {e}")

    thread = threading.Thread(target=_speak)
    thread.start()

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image_advanced(roi):
    # 1. Gamma Correction
    roi_bright = adjust_gamma(roi, gamma=1.5)
    
    # 2. Sharpening
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    roi_sharp = cv2.filter2D(roi_bright, -1, kernel)

    # 3. Standard Prep (30x30) - Matches your preferred model
    roi_rgb = cv2.cvtColor(roi_sharp, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(roi_rgb, (30,30))
    img_final = np.expand_dims(img_resized, axis=0) / 255.0
    
    return img_final

def generate_frames():
    global last_announced, prediction_buffer
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success: break
        
        height, width, _ = frame.shape
        box_size = 220
        x1 = int(width/2 - box_size/2)
        y1 = int(height/2 - box_size/2)
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size != 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_roi, 1.1, 4)
            
            if len(faces) > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "IGNORING FACE", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                prediction_buffer.clear()
            else:
                try:
                    img_final = preprocess_image_advanced(roi)
                    prediction = model.predict(img_final, verbose=0)
                    confidence = np.max(prediction)
                    class_id = np.argmax(prediction)

                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_buffer.append(class_id)
                    else:
                        if len(prediction_buffer) > 0: prediction_buffer.popleft()

                    # STABILIZER: Require 3 consistent frames
                    if len(prediction_buffer) >= 3:
                        most_common_id, num_votes = Counter(prediction_buffer).most_common(1)[0]
                        
                        if num_votes >= 3:
                            translated_text = get_sign_text(most_common_id, current_language)
                            english_text = get_sign_text(most_common_id, 'en')
                            
                            # Visuals
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{english_text} ({int(confidence*100)}%)"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Voice Trigger
                            if translated_text != last_announced:
                                speak_sign(translated_text)
                                last_announced = translated_text
                        else:
                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                except Exception as e:
                    print(f"Prediction Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/set_language/<lang_code>')
def set_language(lang_code):
    global current_language, last_announced
    current_language = lang_code
    last_announced = None 
    return jsonify({"status": "success", "language": lang_code})

@application.route('/get_detection')
def get_detection():
    return jsonify(sign=last_announced if last_announced else "System Active")

@application.teardown_appcontext
def cleanup(resp_or_exc):
    gc.collect()

if __name__ == "__main__":
    application.run(debug=False, threaded=True)