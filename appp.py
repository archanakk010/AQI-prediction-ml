from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Re-define the ColumnDropper class used in the pipeline
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols, errors='ignore')

# ✅ Load the trained pipeline
model = joblib.load('xgb_air_quality_model.pkl')

# ✅ Start Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return " XGB Air Quality Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)

        #  Log prediction
        with open("monitoring_log.txt", "a") as f:
            f.write(f"Input: {input_df.to_dict()}, Prediction: {prediction[0]}\n")

        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

#  Run the server
if __name__ == '__main__':
    app.run(debug=False, port=5000)
