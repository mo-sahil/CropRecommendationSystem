from flask import Flask, render_template, request, jsonify, abort
import os
import joblib
import numpy as np
import traceback


APP = Flask(__name__)
MODELS = {}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


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