from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# === Téléchargement et chargement sécurisés ===
from fetch_from_drive import download_model

def safe_load_pickle(filename):
    model_path = os.path.join("models", filename)
    if not os.path.exists(model_path):
        download_model(filename)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# === Initialisation de Flask ===
app = Flask(__name__)
CORS(app)

# === Chargement des modèles et encodeurs avec protection ===
model = safe_load_pickle('modele_maladies.pkl')
scaler = safe_load_pickle('scaler_maladies.pkl')
disease_encoder = safe_load_pickle('label_encoder_maladies.pkl')
soil_encoder = safe_load_pickle('soil_encoder.pkl')

# === Mapping des types de sol pour le frontend ===
soil_mapping = {i: soil_type for i, soil_type in enumerate(soil_encoder.classes_)}

@app.route('/api/soil-types', methods=['GET'])
def get_soil_types():
    """Retourne la liste des types de sols (tableau)"""
    return jsonify(list(soil_encoder.classes_))

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    """Prédit la maladie du sol à partir des caractéristiques reçues."""
    data = request.get_json()

    required_fields = [
        'soil_type', 'soil_pH', 'temperature', 'humidity',
        'organic_carbon', 'sodium', 'potassium', 'nitrogen', 'phosphorus'
    ]
    
    # Vérifie les champs manquants
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Champs manquants : {missing_fields}'}), 400

    try:
        soil_type_index = data['soil_type']
        if not isinstance(soil_type_index, int) or soil_type_index not in soil_mapping:
            return jsonify({'error': 'Index de type de sol invalide. Consultez /api/soil-types.'}), 400

        try:
            features = [
                float(data['soil_pH']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['organic_carbon']),
                float(data['sodium']),
                float(data['potassium']),
                float(data['nitrogen']),
                float(data['phosphorus']),
                float(soil_type_index)
            ]
        except ValueError:
            return jsonify({'error': 'Tous les champs numériques doivent être valides.'}), 400

        scaled_features = scaler.transform([features])
        prediction_index = model.predict(scaled_features)[0]
        maladie = disease_encoder.inverse_transform([prediction_index])[0]

        if maladie.lower() == "flutariose":
            maladie = "Anthracnose"

        return jsonify({
            'prediction': int(prediction_index),
            'maladie': maladie.capitalize()
        })

    except Exception as e:
        return jsonify({'error': f'Erreur interne : {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
