from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load('model.pkl')
# we do the route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"Mensaje ": "Bienvenido a la API de predicción"}), 200

# Path to make predictions using the loaded model
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "El modelo no ha sido cargado."}), 400

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "No se enviaron datos válidos"}), 400

    features = data['features']

    try:
        
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

