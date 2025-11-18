import joblib
import numpy as np
from flask import Flask, request, jsonify

# --- 1. Load the Model and Vectorizer ---
# Ensure these files are in the same directory as this script!
try:
    # Load the trained LightGBM model
    model = joblib.load('lgbm_model.joblib')
    # Load the TF-IDF vectorizer (with bigrams)
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Models and Vectorizer loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model or Vectorizer files not found. Make sure 'lgbm_model.joblib' and 'tfidf_vectorizer.joblib' are in the correct directory.")
    exit()

# Initialize the Flask app
app = Flask(__name__)

# --- 2. Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to make predictions on input text data.
    The expected input format is a JSON object with a 'text' key:
    {"text": "this product is absolutely amazing and worth the price"}
    """
    
    # Get the data posted by the client
    data = request.get_json(force=True)
    
    # Check if the 'text' field is present
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request'}), 400
    
    # Get the raw text input
    raw_text = data['text']
    
    # 3. Preprocessing and Feature Engineering
    
    # The model expects a sparse matrix of features.
    # The Vectorizer is applied to the text, transforming it into the expected features.
    # We wrap raw_text in a list, as the vectorizer expects an iterable of strings.
    features = vectorizer.transform([raw_text])
    
    # 4. Model Prediction
    # Note: LightGBM predicts a continuous score, which we convert to a single float.
    prediction = model.predict(features)[0]
    
    # We can clip the prediction to ensure it's within a sensible range (e.g., 0 to 100)
    # based on the scale of your target variable. Assuming 0-100 here.
    prediction_clipped = np.clip(prediction, 0, 100).item() # .item() converts numpy float to standard Python float

    # 5. Return the Result
    return jsonify({
        'prediction': round(prediction_clipped, 2),
        'model_used': 'LightGBM',
        'input_text': raw_text
    })

# --- 3. Run the App ---
if __name__ == '__main__':
    # Running the app locally on port 9699 (a common port for deployment)
    app.run(debug=True, host='0.0.0.0', port=9699)
