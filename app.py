from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained XGBoost model
model = joblib.load('Grid_xgb_model.pkl')

# Create the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Air Quality Prediction API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input JSON data
        data = request.get_json(force=True)
        
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # Predict using the model
        prediction = model.predict(input_df)
        
        # Return the prediction
        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, port=5000)
