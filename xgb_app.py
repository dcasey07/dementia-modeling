from flask import Flask, request, jsonify
import pandas as pd
import joblib
import boto3
import numpy as np
import json

app = Flask(__name__)

# Load scaler at app startup
scaler = joblib.load("X_scaler.pkl")

# Initialize boto3 client for SageMaker runtime
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Endpoint name from SageMaker
endpoint_name = 'sagemaker-xgboost-2024-02-29-19-48-11-682'

# Define expected columns after pd.get_dummies()
expected_columns = ['AlcoholLevel',
 'HeartRate',
 'BloodOxygenLevel',
 'BodyTemperature',
 'Weight',
 'MRI_Delay',
 'Age',
 'Dominant_Hand',
 'Gender',
 'Family_History',
 'APOE_ε4',
 'Depression_Status',
 'Cognitive_Test_Scores',
 'Medication_History',
 'Sleep_Quality',
 'Education_Level_Diploma/Degree',
 'Education_Level_No School',
 'Education_Level_Primary School',
 'Education_Level_Secondary School',
 'Smoking_Status_Current Smoker',
 'Smoking_Status_Former Smoker',
 'Smoking_Status_Never Smoked',
 'Physical_Activity_Mild Activity',
 'Physical_Activity_Moderate Activity',
 'Physical_Activity_Sedentary',
 'Nutrition_Diet_Balanced Diet',
 'Nutrition_Diet_Low-Carb Diet',
 'Nutrition_Diet_Mediterranean Diet',
 'Chronic_Health_Conditions_Diabetes',
 'Chronic_Health_Conditions_Heart Disease',
 'Chronic_Health_Conditions_Hypertension',
 'Chronic_Health_Conditions_None']

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from POST request
    data = request.json
    
    # Preprocess the input data
    libsvm_data = preprocess_input_webform(data)
    
    # Sending the LIBSVM data to the SageMaker endpoint for prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/x-libsvm',
        Body=libsvm_data
    )
    
    # Process the response to extract the model's predictions
    result = json.loads(response['Body'].read().decode())
    
    return jsonify(result)

def preprocess_input_webform(data):
    # Convert incoming dictionary to DataFrame
    df_sample = pd.DataFrame([data])
    
    # Apply binary mappings directly in the dataframe
    binary_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'APOE_ε4': {'Positive': 1, 'Negative': 0},
        'Depression_Status': {'Yes': 1, 'No': 0},
        'Sleep_Quality': {'Good': 1, 'Poor': 0},
        'Dominant_Hand': {'Left': 1, 'Right': 0},
        'Family_History': {'Yes': 1, 'No': 0},
        'Medication_History': {'Yes': 1, 'No': 0},
    }
    for column, mapping in binary_mappings.items():
        if column in df_sample.columns:
            df_sample[column] = df_sample[column].map(mapping)
    
    # Convert categorical variables into dummy/indicator variables
    df_sample = pd.get_dummies(df_sample)
    
    # Ensure all expected columns are present, filling missing ones with 0s
    for col in expected_columns:
        if col not in df_sample.columns:
            df_sample[col] = 0

    # Reorder columns to match the training data
    df_sample = df_sample[expected_columns]
    
    # Scale the features
    scaled_features = scaler.transform(df_sample)
    
    # Convert scaled features to LIBSVM format for a single sample
    libsvm_str = to_libsvm(scaled_features)
    
    return libsvm_str

def to_libsvm(features):
    """Convert a single sample's features to LIBSVM string format."""
    libsvm_str = "0"  # Placeholder for the target variable, assuming it's not included in the features
    for i, value in enumerate(features.flatten(), start=1):
        if value != 0:  # Only include non-zero features
            libsvm_str += f" {i}:{value}"
    return libsvm_str

if __name__ == '__main__':
    app.run(debug=True)