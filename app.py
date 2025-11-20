from flask import Flask, render_template, request, jsonify, abort
import os
import joblib
import numpy as np
import traceback


APP = Flask(__name__)
MODELS = {}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

from flask import Flask, render_template, request, jsonify, abort, send_file
import os
import joblib
import numpy as np
import traceback
import os
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO

APP = Flask(__name__)
MODELS = {}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

os.environ["GOOGLE_API_KEY"] = "AIzaSyDuCiZoRka1euEpvMgIoy-m7cRoWWjUbBQ"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

gemini_model = genai.GenerativeModel('gemini-2.5-flash')


def load_models():
    global MODELS
    MODELS = {}
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.pkl'):
            path = os.path.join(MODELS_DIR, fname)
            try:
                model = joblib.load(path)
                MODELS[fname] = model
                print(f"Loaded model: {fname}")
            except Exception as e:
                print(f"Failed to load {fname}: {e}")




@APP.route('/')
def index():
    return render_template('index.html', models=sorted(MODELS.keys()), features=FEATURE_NAMES)

@APP.route('/explain', methods=['POST'])
def explain():
    data = request.json
    features = data.get('features')
    prediction = data.get('prediction')
    language = data.get('language', 'English')

    prompt = f"""
    Act as an agricultural expert. 
    A machine learning model predicted '{prediction}' (probability: High) 
    is the best crop for these soil conditions: {features}.
    
    Explain in 3 simple bullet points why '{prediction}' is suitable here.
    
    IMPORTANT: Provide the response strictly in {language} language.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"explanation": response.text})
    except Exception as e:
        return jsonify({"explanation": f"AI Error: {str(e)}"}), 500

@APP.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        if not data:
            print("Error: No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        text = data.get('text')
        lang_name = data.get('language', 'English')

        # Debug print: Check what your server is actually receiving
        print(f"TTS Request -> Language: {lang_name} | Text length: {len(str(text))}")

        if not text or not text.strip():
            print("Error: Text is empty!")
            return jsonify({"error": "No text provided to speak"}), 400

        # Map full language names to gTTS codes
        lang_map = {
            'English': 'en', 'Hindi': 'hi', 'Urdu': 'ur', 'Spanish': 'es',
            'French': 'fr', 'Bengali': 'bn', 'Marathi': 'mr', 'Tamil': 'ta', 'Telugu': 'te'
        }
        
        lang_code = lang_map.get(lang_name, 'en')

        # Generate Audio
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        return send_file(audio_io, mimetype='audio/mpeg')

    except Exception as e:
        # This print statement is CRITICAL. It will show the real error in your terminal.
        print(f"❌ TTS SERVER ERROR: {str(e)}")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500

 
@APP.route('/models')
def models_list():
    return jsonify({'models': sorted(MODELS.keys())})




def parse_features_from_dict(d):
    vals = []
    for f in FEATURE_NAMES:
        if f not in d:
            raise KeyError(f"Missing feature: {f}")
        vals.append(float(d[f]))
    return np.array(vals).reshape(1, -1)




@APP.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400


    model_name = data.get('model')
    features = data.get('features')
    if not model_name or not features:
        return jsonify({'error': 'Provide "model" and "features" in JSON'}), 400


    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 404


    model = MODELS[model_name]
    try:
        X = parse_features_from_dict(features)
        pred = model.predict(X)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X).tolist()
        result = {
            'model': model_name,
            'prediction': pred.tolist(),
            'probabilities': proba
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




@APP.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from form
    data = request.form.to_dict()
    
    # Extract features in the correct order expected by your model
    # (Ensure these match your training columns exactly)
    feature_values = [
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]
    features = np.array([feature_values])

    # 2. Load the specific ML model requested by the user
    selected_model_name = data.get('model')
    
    # --- CRITICAL STEP: Load your ML model here ---
    # Adjust this line to match how you store your models. 
    # Example: loading from a file based on the name
    ml_model = joblib.load(f"models/{selected_model_name}") 
    
    # 3. Predict Probabilities using the ML model
    # We use 'ml_model' here, not the global 'gemini_model'
    probs = ml_model.predict_proba(features)[0]
    class_names = ml_model.classes_

    # 4. Sort and format results
    sorted_probs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    
    top_5 = []
    for crop, score in sorted_probs[:5]:
        top_5.append({
            "crop": crop,
            "probability": round(score * 100, 2)
        })

    return jsonify({
        "top_prediction": top_5[0]['crop'],
        "all_predictions": top_5
    })




if __name__ == '__main__':
    load_models()
    APP.run(host='0.0.0.0', port=5000, debug=True)
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']




def load_models():
    global MODELS
    MODELS = {}
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.pkl'):
            path = os.path.join(MODELS_DIR, fname)
            try:
                model = joblib.load(path)
                MODELS[fname] = model
                print(f"Loaded model: {fname}")
            except Exception as e:
                print(f"Failed to load {fname}: {e}")




@APP.route('/')
def index():
    return render_template('index.html', models=sorted(MODELS.keys()), features=FEATURE_NAMES)




@APP.route('/models')
def models_list():
    return jsonify({'models': sorted(MODELS.keys())})




def parse_features_from_dict(d):
    vals = []
    for f in FEATURE_NAMES:
        if f not in d:
            raise KeyError(f"Missing feature: {f}")
        vals.append(float(d[f]))
    return np.array(vals).reshape(1, -1)




@APP.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400


    model_name = data.get('model')
    features = data.get('features')
    if not model_name or not features:
        return jsonify({'error': 'Provide "model" and "features" in JSON'}), 400


    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 404


    model = MODELS[model_name]
    try:
        X = parse_features_from_dict(features)
        pred = model.predict(X)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X).tolist()
        result = {
            'model': model_name,
            'prediction': pred.tolist(),
            'probabilities': proba
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




@APP.route('/predict', methods=['POST'])
def form_predict():
    form = request.form
    model_name = form.get('model')
    if not model_name:
        return jsonify({'error': 'No model selected'}), 400
    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 404


    try:
        features = {f: form.get(f) for f in FEATURE_NAMES}
        X = parse_features_from_dict(features)
        model = MODELS[model_name]
        pred = model.predict(X)
        proba = model.predict_proba(X).tolist() if hasattr(model, 'predict_proba') else None
        return jsonify({'model': model_name, 'prediction': pred.tolist(), 'probabilities': proba})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    load_models()
    APP.run(host='0.0.0.0', port=5000, debug=True)