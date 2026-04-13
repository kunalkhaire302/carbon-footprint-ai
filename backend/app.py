from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import subprocess
from utils import predict_emission, compute_breakdown, generate_suggestions, calculate_percentile_and_grade, INDIA_AVG, WORLD_AVG

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Predict total emission
        total_co2 = predict_emission(data)
        
        # Breakdown into categories
        breakdown = compute_breakdown(data)
        
        # Calculate grade
        percentile, grade = calculate_percentile_and_grade(total_co2)
        
        # Generate suggestions
        suggestions = generate_suggestions(breakdown)

        response = {
            "total_footprint_tco2e": total_co2,
            "category_breakdown": breakdown,
            "comparison": {
                "india_avg": INDIA_AVG,
                "world_avg": WORLD_AVG,
                "your_value": total_co2,
                "percentile": percentile,
                "grade": grade
            },
            "suggestions": suggestions
        }

        # Save to history
        history = load_history()
        history.insert(0, {
            "input": data,
            "prediction": response
        })
        history = history[:10]  # Keep last 10
        save_history(history)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(load_history())

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Trigger retraining pipeline
        process = subprocess.Popen(["python", "backend/model.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return jsonify({"error": "Retraining failed", "details": stderr.decode('utf-8')}), 500
            
        # Try reloading the model & preprocessor (Note: in a real WSGI setup this might require restarting workers)
        import joblib
        import utils
        utils.preprocessor = joblib.load('model/preprocessor.pkl')
        utils.model = joblib.load('model/carbon_model.pkl')
        
        return jsonify({"message": "Retraining completed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
