import os
import cv2
import base64
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from gtts import gTTS
from languages import get_sign_text

# Force CPU mode to save memory on Render
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

application = Flask(__name__)

# --- GLOBAL STATE ---
last_announced = "System Active"
current_language = 'en'

# --- MODEL LOADING ---
try:
    # Standard Keras 3 loading
    model = load_model('models/road_sign_model_final.h5')
    print("✅ SUCCESS: Model loaded with Keras 3")
except Exception as e:
    print(f"❌ ERROR: Model load failed: {e}")
    model = None

def preprocess_frame(frame):
    # Convert BGR (Camera) to RGB (AI Model)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (30, 30))
    # Normalize pixels to 0-1 range
    return np.expand_dims(resized, axis=0).astype('float32') / 255.0

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global last_announced, current_language
    if model is None: return jsonify(success=False, error="Model offline")
    
    try:
        data = request.get_json()
        # Decode the image sent from your iPhone/Laptop
        raw_b64 = data['image'].split(',')[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # AI Prediction
        inp = preprocess_frame(frame)
        pred = model.predict(inp, verbose=0)
        conf = np.max(pred)
        
        # Lowered threshold to 0.50 for easier demo recognition
        if conf > 0.50:
            class_id = np.argmax(pred)
            sign_text = get_sign_text(class_id, current_language)
            
            if sign_text != last_announced:
                last_announced = sign_text
                # Background thread for audio so the video doesn't lag
                def make_audio():
                    tts = gTTS(text=sign_text, lang=current_language)
                    tts.save("static/announcement.mp3")
                threading.Thread(target=make_audio).start()
        
        return jsonify(success=True)
    except Exception as e:
        print(f"Prediction Error: {e}")
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