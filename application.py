import os
# Force Keras 3 behavior
os.environ["TF_USE_LEGACY_KERAS"] = "0" 

import cv2
import base64
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from gtts import gTTS
from languages import get_sign_text

application = Flask(__name__)

# --- GLOBAL STATE ---
last_announced = "System Active"
current_language = 'en'

# --- THE FIX: LOADING ---
try:
    # We use compile=False to bypass training-specific errors during loading
    model = load_model('models/road_sign_model_final.h5', compile=False)
    print("✅ MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print(f"❌ FINAL LOAD ERROR: {e}")
    model = None

def preprocess(frame):
    # RGB Conversion is vital for GTSRB models
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (30, 30))
    return np.expand_dims(resized, axis=0).astype('float32') / 255.0

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global last_announced, current_language
    if model is None: return jsonify(success=False)
    try:
        data = request.get_json()
        raw_b64 = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        inp = preprocess(frame)
        pred = model.predict(inp, verbose=0)
        conf = np.max(pred)
        class_id = np.argmax(pred)

        # Monitor this in Render Logs!
        print(f"Prediction: Class {class_id} ({conf*100:.1f}%)")

        if conf > 0.45:
            sign_text = get_sign_text(class_id, current_language)
            if sign_text != last_announced:
                last_announced = sign_text
                def speak():
                    tts = gTTS(text=sign_text, lang=current_language)
                    tts.save("static/announcement.mp3")
                threading.Thread(target=speak).start()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False)

@application.route('/get_detection')
def get_detection():
    return jsonify(sign=last_announced)

@application.route('/set_language/<lang>')
def set_language(lang):
    global current_language
    current_language = lang
    return jsonify(status="ok")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host='0.0.0.0', port=port)