import streamlit as st
import requests
import pandas as pd
import json
import os

# --- Configuration ---
# CORRECTED: The API runs on port 8000 by default with the uvicorn command
API_BASE_URL = "http://127.0.0.1:8000" 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'training_dataset.json')


# --- Main App UI ---
st.set_page_config(layout="wide")
st.title("Recruitment Model MVP Interface")
st.markdown("A simple interface to interact with the recruitment matching API.")

# --- Sidebar for Actions ---
st.sidebar.header("Model Actions")

# 1. Button to Train a New Model
if st.sidebar.button("Train New Model"):
    with st.spinner("Training in progress... This may take a few minutes."):
        try:
            response = requests.post(f"{API_BASE_URL}/train")
            if response.status_code == 201:
                st.sidebar.success(f"New model trained successfully! File: {response.json().get('new_model_file')}")
            else:
                st.sidebar.error(f"Training failed: {response.text}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"API connection error: {e}")

# --- Main Page Layout ---

# Section 1: Model Selection and Evaluation
st.header("1. Evaluate Model Performance")
col1, col2 = st.columns(2)

with col1:
    try:
        models_response = requests.get(f"{API_BASE_URL}/models")
        if models_response.status_code == 200:
            available_models = models_response.json().get("models", [])
            selected_model = st.selectbox("Choose a model to evaluate:", available_models)
        else:
            st.warning("Could not fetch model list. Is the API running?")
            selected_model = None
    except requests.exceptions.RequestException:
        st.error("API connection error. Please ensure the backend is running.")
        selected_model = None

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    if selected_model and st.button("Evaluate Selected Model"):
        with st.spinner(f"Evaluating {selected_model}..."):
            eval_response = requests.get(f"{API_BASE_URL}/evaluate/{selected_model}")
            if eval_response.status_code == 200:
                metrics = eval_response.json()
                st.subheader(f"Performance for `{metrics['model_filename']}`")
                st.metric("Accuracy", metrics['accuracy'])
                st.text("Classification Report:")
                # Convert dict to a more readable format
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df)
            else:
                st.error(f"Evaluation failed: {eval_response.text}")


st.divider()

# Section 2: Predict Applicant Matches
st.header("2. Find Best Vacancy Matches for an Applicant")

# Text area for raw applicant JSON
st.subheader("Paste Applicant's Raw JSON here:")
default_applicant_json = {
    "codigo_profissional": "24645-example",
    "cv_pt": "analista de teste/qa... profissional hands on... automação de testes... selenium webdriver + java..."
}
applicant_json_str = st.text_area("Applicant Data", json.dumps(default_applicant_json, indent=4), height=250)

if st.button("Match Applicant to Vacancies"):
    if not applicant_json_str:
        st.error("Please paste the applicant's JSON data.")
    else:
        try:
            applicant_data = json.loads(applicant_json_str)
            with st.spinner("Finding the best matches..."):
                # We use the 'latest' model for this prediction
                predict_response = requests.post(f"{API_BASE_URL}/predict/latest", json=applicant_data)

                if predict_response.status_code == 200:
                    results = predict_response.json()
                    st.success(f"Top 5 Matches found using model: `{results['model_used']}`")
                    
                    results_df = pd.DataFrame(results['top_matches'])
                    # Format the probability as a percentage
                    results_df['match_probability'] = results_df['match_probability'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {predict_response.text}")

        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check the pasted data.")
        except requests.exceptions.RequestException as e:
            st.error(f"API connection error: {e}")

st.divider()

# Section 3: Browse Training Data
st.header("3. Browse the Training Dataset")
if st.button("Load and Show Training Data"):
    try:
        df_training = pd.read_json(TRAINING_DATA_PATH)
        st.dataframe(df_training)
    except FileNotFoundError:
        st.error(f"Could not find the training dataset at {TRAINING_DATA_PATH}")