import os
import cv2
import base64
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from gtts import gTTS
from languages import get_sign_text

application = Flask(__name__)

# --- CONFIG ---
last_announced = "System Active"
current_language = 'en'

# Load Model - Modern Keras 3 way
try:
    model = load_model('models/road_sign_model_final.h5')
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Load Error: {e}")
    model = None

def preprocess(frame):
    # Convert BGR (OpenCV) to RGB (Model)
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
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Predict
        inp = preprocess(frame)
        pred = model.predict(inp, verbose=0)
        conf = np.max(pred)
        
        if conf > 0.60: # Threshold
            class_id = np.argmax(pred)
            sign_text = get_sign_text(class_id, current_language)
            if sign_text != last_announced:
                last_announced = sign_text
                # Generate audio
                tts = gTTS(text=sign_text, lang=current_language)
                tts.save("static/announcement.mp3")
        
        return jsonify(success=True)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False)

@application.route('/get_detection')
def get_detection():
    global last_announced
    return jsonify(sign=last_announced)

@application.route('/set_language/<lang>')
def set_language(lang):
    global current_language
    current_language = lang
    return jsonify(status="success")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host='0.0.0.0', port=port)