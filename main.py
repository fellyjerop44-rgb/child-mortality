import streamlit as st
import joblib
import numpy as np
import pandas as pd # Import pandas for data manipulation

st.title("Mortality Prediction App")

st.write("Loading model...")

try:
    # Define the path to your saved Random Forest model
    model_filename = "project_random_forest.joblib"
    model = joblib.load(model_filename)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load the model from {model_filename}.")
    st.exception(e)
    st.stop()

st.write("This app predicts Mortality based on Period and Causes.")

# --- Input fields for rf_model compatible features ---

# Input for Period
period = st.number_input(
    "Period (Year)",
    min_value=2000,
    max_value=2025,
    value=2020,
    step=1
)

# List of all possible causes (must match the categories used during training)
# This list includes 'Congenital anomalies' even if it was dropped during one-hot encoding
# because the user needs to select from all possible categories.
all_causes = [
    'Congenital anomalies',
    'Diarrhoeal diseases',
    'HIV/AIDS',
    'Malaria',
    'Measles',
    'Meningitis/encephalitis',
    'Prematurity',
    'Sepsis and other infectious conditions of the newborn',
    'Tetanus',
    'Tuberculosis'
]

selected_cause = st.selectbox(
    "Select Cause:",
    options=all_causes
)


if st.button("Predict"):    
    try:
        # Define the exact column names and order as used in X_train during model training
        # This is CRUCIAL for the model to make correct predictions.
        feature_names_from_xtrain = [
            'Period',
            'Causes_Diarrhoeal diseases',
            'Causes_HIV/AIDS',
            'Causes_Malaria',
            'Causes_Measles',
            'Causes_Meningitis/encephalitis',
            'Causes_Prematurity',
            'Causes_Sepsis and other infectious conditions of the newborn',
            'Causes_Tetanus',
            'Causes_Tuberculosis'
        ]

        # Create a DataFrame for the input features, initialized with zeros
        input_data_for_prediction = pd.DataFrame(0, index=[0], columns=feature_names_from_xtrain)
        
        # Set the 'Period' value
        input_data_for_prediction['Period'] = period

        # Set the appropriate one-hot encoded 'Causes_' column to 1
        # 'Congenital anomalies' is the reference category (dropped_first=True), so its columns are all zeros.
        if selected_cause != 'Congenital anomalies':
            one_hot_col_name = f'Causes_{selected_cause}'
            if one_hot_col_name in input_data_for_prediction.columns: # Check if the column exists in our expected features
                input_data_for_prediction[one_hot_col_name] = 1
            else:
                st.warning(f"Selected cause '{selected_cause}' does not have a corresponding feature column in the model. This might lead to incorrect predictions.")

        # Make prediction
        prediction = model.predict(input_data_for_prediction)

        st.success(f"Predicted Mortality: {prediction[0]:.2f}")

    except Exception as e:
        st.error("Prediction failed. Please check inputs and model compatibility.")
        st.exception(e)
