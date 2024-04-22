from flask import Flask, request, jsonify, send_file
from utilities import predict
from pymongo import MongoClient
import os

app = Flask(__name__)

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['predictions']
collection = db['prediction_data']

# Define a route to serve the front end
@app.route('/')
def index():
    return send_file('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def prediction():
    # Get data from the request
    data = request.get_json()

    # Check if 'context' and 'claim' are present in the request
    if 'context' not in data or 'claim' not in data:
        return jsonify({'error': 'Missing required fields: context and claim'}), 400

    # Get context and claim from the request
    context = data['context']
    claim = data['claim']

    # Perform prediction using the predict function from utilities.py
    result = predict(context, claim)

    # Save the prediction data and input to MongoDB
    prediction_entry = {
        'context': context,
        'claim': claim,
        'prediction': {
            'predicted_label': result['predicted_label'],
            'evidence': result['evidence'],
            'probabilities': result['probabilities']
        }
    }
    collection.insert_one(prediction_entry)

    # Return the prediction result as JSON
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False)
