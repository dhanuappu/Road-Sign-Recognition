import os
import gc 
import cv2
import base64
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from gtts import gTTS
from collections import deque, Counter
from languages import get_sign_text

# --- STABILITY: CPU MODE ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

application = Flask(__name__)

# --- GLOBAL VARIABLES ---
last_announced = "System Active"
current_language = 'en'
prediction_buffer = deque(maxlen=5)

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.50 # Lowered for demo stability

# Load Model
try:
    model = load_model('models/road_sign_model_final.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image(roi):
    # 1. Convert BGR to RGB (Fixes the "Wrong Color" issue)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # 2. Brightness correction
    roi_bright = adjust_gamma(roi_rgb, gamma=1.5)
    # 3. Resize to 30x30
    img_resized = cv2.resize(roi_bright, (30,30))
    # 4. Normalize (0 to 1) and expand dims
    img_final = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0
    return img_final

def generate_audio_file(text, lang):
    try:
        if not os.path.exists('static'): os.makedirs('static')
        tts = gTTS(text=text, lang=lang)
        tts.save("static/announcement.mp3")
    except Exception as e:
        print(f"Audio Error: {e}")

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global last_announced, current_language, prediction_buffer
    if model is None: return jsonify(success=False, error="Model offline")

    try:
        data = request.get_json()
        # Extract raw base64 data
        image_b64 = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Skip if face detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(face_cascade.detectMultiScale(gray, 1.1, 4)) > 0:
            return jsonify(success=True, msg="Ignoring Face")

        # Prediction logic
        processed = preprocess_image(frame)
        prediction = model.predict(processed, verbose=0)
        conf = np.max(prediction)
        class_id = np.argmax(prediction)

        # DEBUG: Check Render Logs for this line!
        print(f"DEBUG: Class {class_id} | Conf {conf*100:.1f}%")

        if conf > CONFIDENCE_THRESHOLD:
            sign_text = get_sign_text(class_id, current_language)
            if sign_text != last_announced:
                last_announced = sign_text
                threading.Thread(target=generate_audio_file, args=(sign_text, current_language)).start()
        
        return jsonify(success=True)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify(success=False)

@application.route('/get_detection')
def get_detection():
    global last_announced
    return jsonify(sign=last_announced)

@application.route('/set_language/<lang_code>')
def set_language(lang_code):
    global current_language, last_announced
    current_language = lang_code
    last_announced = "Language Updated"
    return jsonify(status="success")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host='0.0.0.0', port=port)