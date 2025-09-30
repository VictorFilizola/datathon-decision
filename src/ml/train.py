import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'training_dataset.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def run_training_pipeline():
    """
    Loads the training data, trains a classifier, evaluates it, and saves the versioned model.
    """
    print("--- Starting Model Training & Evaluation Pipeline ---")

    # --- 1. Load Dataset ---
    try:
        df = pd.read_json(DATASET_PATH)
        print(f"-> Successfully loaded training dataset with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: The file {DATASET_PATH} was not found.")
        print("Please run 'create_training_data.py' first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    if df.empty:
        print("The training dataset is empty. Aborting training.")
        return

    # --- 2. Define Features and Target ---
    features = [
        'skill_match_score',
        'level_match_score',
        'applicant_skills_count',
        'vacancy_skills_count'
    ]
    target = 'hired'

    X = df[features]
    y = df[target]

    print(f"\nFeatures being used for training: {features}")
    print(f"Target variable: '{target}'")

    # --- 3. Split Data into Training and Testing Sets ---
    # Using stratify=y helps ensure the test set has a similar proportion of hired/not hired as the training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Data split into training ({len(X_train)} records) and testing ({len(X_test)} records) sets.")

    # --- 4. Train the Model ---
    print("\n-> Training the RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("-> Model training complete.")

    # --- 5. Make Predictions and Evaluate ---
    print("\n--- Model Performance Evaluation ---")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2%}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("(Rows: Actual Class, Columns: Predicted Class)\n")
    
    print("Classification Report:")
    # Added zero_division=0 to handle cases where a class has no predictions in the test set
    report = classification_report(y_test, predictions, target_names=['Not Hired', 'Hired'], zero_division=0)
    print(report)
    print("------------------------------------")


    # --- 6. Save the Trained and Versioned Model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"recruitment_model_{timestamp}.joblib"
    model_output_path = os.path.join(MODEL_DIR, model_filename)
    
    joblib.dump(model, model_output_path)
    print(f"\n-> Trained model successfully saved to: {model_output_path}")

    print("\n--- Model Training Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    run_training_pipeline()