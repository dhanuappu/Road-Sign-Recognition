import os
import gc 
import cv2
import base64
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf_keras as keras
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
CONFIDENCE_THRESHOLD = 0.85 

# Load Model
try:
    model = load_model('models/road_sign_model_final.h5')
    print("✅ Model loaded successfully using Keras 3!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image_advanced(roi):
    # 1. Gamma Correction for better visibility
    roi_bright = adjust_gamma(roi, gamma=1.5)
    # 2. Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    roi_sharp = cv2.filter2D(roi_bright, -1, kernel)
    # 3. Standard Prep (30x30)
    roi_rgb = cv2.cvtColor(roi_sharp, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(roi_rgb, (30,30))
    img_final = np.expand_dims(img_resized, axis=0) / 255.0
    return img_final

def generate_audio_file(text, lang):
    """Generates an MP3 file using gTTS for the browser to play."""
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("static/announcement.mp3")
    except Exception as e:
        print(f"Audio generation error: {e}")

# --- ROUTES ---

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global last_announced, current_language, prediction_buffer
    
    if model is None:
        return jsonify(success=False, error="Model not loaded")

    try:
        # 1. Get Base64 image from JavaScript
        data = request.get_json()
        image_b64 = data['image'].split(',')[1]
        
        # 2. Decode to OpenCV format
        nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Detect Faces (to ignore them)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            prediction_buffer.clear()
            return jsonify(success=True, sign="IGNORING FACE")

        # 4. Predict Road Sign
        img_final = preprocess_image_advanced(frame)
        prediction = model.predict(img_final, verbose=0)
        confidence = np.max(prediction)
        class_id = np.argmax(prediction)

        if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(class_id)
            
            # STABILIZER: Require consistent frames
            if len(prediction_buffer) >= 3:
                most_common_id, num_votes = Counter(prediction_buffer).most_common(1)[0]
                if num_votes >= 3:
                    translated_text = get_sign_text(most_common_id, current_language)
                    
                    if translated_text != last_announced:
                        last_announced = translated_text
                        # Generate audio in a background thread
                        threading.Thread(target=generate_audio_file, args=(translated_text, current_language)).start()

        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@application.route('/get_detection')
def get_detection():
    global last_announced
    # This solves your 'last_announced' NameError
    return jsonify(sign=last_announced if last_announced else "System Active")

@application.route('/set_language/<lang_code>')
def set_language(lang_code):
    global current_language, last_announced
    current_language = lang_code
    last_announced = None 
    return jsonify({"status": "success", "language": lang_code})

@application.teardown_appcontext
def cleanup(resp_or_exc):
    gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host='0.0.0.0', port=port)