import os

# Reduce TensorFlow console messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

# Reduce TensorFlow and absl logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Hide sklearn version compatibility warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL, LABEL ENCODER AND SCALER
# ============================================================

model = tf.keras.models.load_model(
    "models/fertilizer.h5",
    compile=False
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# ============================================================
# CSV FILE PATH
# ============================================================

csv_file = r"C:\514\PROJECTS\fertilizerrecommendation\data\f2.csv"


# ============================================================
# FUNCTION TO HANDLE UNSEEN LABELS
# ============================================================

def encode_label(value, encoder):

    if value in encoder.classes_:
        return encoder.transform([value])[0]

    else:
        # Add unseen label to known labels
        encoder.classes_ = np.append(
            encoder.classes_,
            value
        )

        return encoder.transform([value])[0]


# ============================================================
# FERTILIZER PREDICTION FUNCTION
# ============================================================

def predict_fertilizer(
    soil_type,
    crop_type,
    nitrogen,
    phosphorous,
    potassium
):

    # Expected number of features used by the trained model
    expected_features = 25

    # Encode categorical inputs
    soil_encoded = encode_label(
        soil_type,
        label_encoder
    )

    crop_encoded = encode_label(
        crop_type,
        label_encoder
    )

    # Create input feature array
    input_data = np.array([
        [
            soil_encoded,
            crop_encoded,
            nitrogen,
            phosphorous,
            potassium
        ]
    ])

    # Add missing features as zeros
    if input_data.shape[1] < expected_features:

        padding = np.zeros(
            (
                input_data.shape[0],
                expected_features - input_data.shape[1]
            )
        )

        input_data = np.hstack(
            (
                input_data,
                padding
            )
        )

    # Scale input data
    input_data_scaled = scaler.transform(
        input_data
    )

    # Make prediction
    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    # Decode predicted class
    fertilizer_name = label_encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    return fertilizer_name


# ============================================================
# HOME PAGE
# ============================================================

@app.route('/')
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get values from form
        soil_type = request.form[
            'soil_type'
        ].strip().lower()

        crop_type = request.form[
            'crop_type'
        ].strip().lower()

        nitrogen = float(
            request.form['nitrogen']
        )

        potassium = float(
            request.form['potassium']
        )

        phosphorous = float(
            request.form['phosphorous']
        )


        # ====================================================
        # LOAD CSV
        # ====================================================

        df = pd.read_csv(
            csv_file
        )


        # ====================================================
        # CONVERT NUMERIC COLUMNS TO FLOAT
        # ====================================================

        df['Nitrogen'] = df[
            'Nitrogen'
        ].astype(float)

        df['Potassium'] = df[
            'Potassium'
        ].astype(float)

        df['Phosphorous'] = df[
            'Phosphorous'
        ].astype(float)


        # ====================================================
        # NORMALIZE TEXT COLUMNS
        # ====================================================

        df['Soil_Type'] = df[
            'Soil_Type'
        ].str.strip().str.lower()

        df['Crop_Type'] = df[
            'Crop_Type'
        ].str.strip().str.lower()


        # ====================================================
        # CHECK WHETHER INPUT ALREADY EXISTS IN CSV
        # ====================================================

        match = df[
            (df['Soil_Type'] == soil_type) &
            (df['Crop_Type'] == crop_type) &
            (df['Nitrogen'] == nitrogen) &
            (df['Potassium'] == potassium) &
            (df['Phosphorous'] == phosphorous)
        ]


        # ====================================================
        # USE EXISTING FERTILIZER IF FOUND
        # ====================================================

        if not match.empty:

            fertilizer = match.iloc[0][
                'Fertilizer'
            ]


        # ====================================================
        # OTHERWISE USE ML MODEL
        # ====================================================

        else:

            fertilizer = predict_fertilizer(
                soil_type,
                crop_type,
                nitrogen,
                phosphorous,
                potassium
            )


            # =================================================
            # ADD NEW PREDICTION TO CSV
            # =================================================

            new_data = pd.DataFrame(
                [
                    [
                        soil_type,
                        crop_type,
                        nitrogen,
                        phosphorous,
                        potassium,
                        fertilizer
                    ]
                ],
                columns=[
                    'Soil_Type',
                    'Crop_Type',
                    'Nitrogen',
                    'Phosphorous',
                    'Potassium',
                    'Fertilizer'
                ]
            )


            df = pd.concat(
                [
                    df,
                    new_data
                ],
                ignore_index=True
            )


            df.to_csv(
                csv_file,
                index=False
            )


        # ====================================================
        # RETURN PREDICTION
        # ====================================================

        return jsonify({
            "prediction": fertilizer
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=False
    )