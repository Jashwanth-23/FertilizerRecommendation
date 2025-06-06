import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR


app = Flask(__name__)

# Load the trained model, label encoder, and scaler
# model = tf.keras.models.load_model(r"C:\Users\Jashwanth Chepuri\OneDrive\Desktop\Fr\fertilizerrecommendation\models\fertilizer.h5")
# label_encoder = joblib.load(r"C:\Users\Jashwanth Chepuri\OneDrive\Desktop\Fr\fertilizerrecommendation\models\label_encoder.pkl")
# scaler = joblib.load(r"C:\Users\Jashwanth Chepuri\OneDrive\Desktop\Fr\fertilizerrecommendation\models\scaler.pkl")

# Load the CSV file
csv_file = r"C:\Users\Jashwanth Chepuri\OneDrive\Desktop\Fr\fertilizerrecommendation\data\f2.csv"

# Function to handle unseen labels
def encode_label(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Add unseen label to known labels
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

def predict_fertilizer(soil_type, crop_type, nitrogen, phosphorous, potassium):
    # Load the original training columns (optional, if known)
    expected_features = 25  # Update this based on your training data

    # Encode categorical inputs
    soil_encoded = encode_label(soil_type, label_encoder) # type: ignore
    crop_encoded = encode_label(crop_type, label_encoder) # type: ignore

    # Create input feature array
    input_data = np.array([[soil_encoded, crop_encoded, nitrogen, phosphorous, potassium]])

    # Ensure the input data has the expected shape (add missing features as zeros if needed)
    if input_data.shape[1] < expected_features:
        padding = np.zeros((input_data.shape[0], expected_features - input_data.shape[1]))
        input_data = np.hstack((input_data, padding))

    # Scale input data
    input_data_scaled = scaler.transform(input_data) # type: ignore

    # Make prediction
    prediction = model.predict(input_data_scaled) # type: ignore

    # Decode the predicted class
    fertilizer_name = label_encoder.inverse_transform([np.argmax(prediction)])[0] # type: ignore

    return fertilizer_name


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        soil_type = request.form['soil_type'].strip().lower()
        crop_type = request.form['crop_type'].strip().lower()
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])
        

        # Load the CSV
        df = pd.read_csv(r"C:\Users\Jashwanth Chepuri\OneDrive\Desktop\Fr\fertilizerrecommendation\data\f2.csv")

        # Ensure numeric values in CSV are floats
        df['Nitrogen'] = df['Nitrogen'].astype(float)
        df['Potassium'] = df['Potassium'].astype(float)
        df['Phosphorous'] = df['Phosphorous'].astype(float)
        

        # Normalize text columns (lowercase for comparison)
        df['Soil_Type'] = df['Soil_Type'].str.strip().str.lower()
        df['Crop_Type'] = df['Crop_Type'].str.strip().str.lower()

        # Check if input already exists in CSV
        match = df[
            (df['Soil_Type'] == soil_type) & 
            (df['Crop_Type'] == crop_type) & 
            (df['Nitrogen'] == nitrogen) & 
            (df['Potassium'] == potassium) & 
            (df['Phosphorous'] == phosphorous)
        ]

        if not match.empty:
            # Return existing fertilizer if found
            fertilizer = match.iloc[0]['Fertilizer']
        else:
            # Predict using the ML model
            fertilizer = predict_fertilizer(soil_type, crop_type, nitrogen, phosphorous, potassium)

            # Update CSV with new data
            new_data = pd.DataFrame([[soil_type, crop_type, nitrogen, phosphorous, potassium, fertilizer]], 
                                    columns=['Soil_Type', 'Crop_Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'Fertilizer'])
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(csv_file, index=False)

        return jsonify({"prediction": fertilizer})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
     app.run(debug=True)
