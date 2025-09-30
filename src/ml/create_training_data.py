import json
import os

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
AGGREGATED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'prospects_aggregated.json')
VACANCIES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'vacancies_enhanced.json')
APPLICANTS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'applicants_enhanced.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'training_dataset.json')

# --- Feature Engineering Functions ---

def calculate_skill_match(applicant_skills, vacancy_skills):
    """Calculates the percentage of vacancy skills that the applicant has."""
    if not vacancy_skills:
        return 1.0  # If no skills are required, it's a perfect match
    if not applicant_skills:
        return 0.0

    match_count = len(set(applicant_skills) & set(vacancy_skills))
    return match_count / len(vacancy_skills)

def calculate_level_match(applicant_level, vacancy_level):
    """Calculates a score for experience level match."""
    levels = ['junior', 'pleno', 'senior', 'leadership']
    if applicant_level == "not specified" or vacancy_level == "not specified":
        return 0.0
    
    try:
        app_idx = levels.index(applicant_level)
        vac_idx = levels.index(vacancy_level)

        if app_idx == vac_idx:
            return 1.0  # Perfect match
        elif app_idx > vac_idx:
            return 0.75 # Overqualified is still a good match
        elif vac_idx - app_idx == 1:
            return 0.5 # Slightly underqualified
        else:
            return 0.0 # Too underqualified
    except ValueError:
        return 0.0 # If a level is not in our list

def run_dataset_creation():
    """
    Creates the final training dataset by combining aggregated prospects
    with enhanced features and engineering new matching features.
    """
    print("--- Starting Training Dataset Creation Pipeline ---")

    # --- 1. Load All Necessary Data ---
    try:
        with open(AGGREGATED_PATH, 'r', encoding='utf-8') as f:
            aggregated_data = json.load(f)
        with open(VACANCIES_PATH, 'r', encoding='utf-8') as f:
            vacancies_enhanced = json.load(f)
        with open(APPLICANTS_PATH, 'r', encoding='utf-8') as f:
            applicants_enhanced = json.load(f)
        print("-> All required data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all processed data files exist before running this script.")
        return

    training_dataset = []

    print("-> Assembling training records and engineering features...")
    # --- 2. Iterate and Build Records ---
    for prospect_entry in aggregated_data:
        vaga_id = prospect_entry.get("vaga_id")
        vacancy_features = vacancies_enhanced.get(vaga_id)

        if not vacancy_features:
            continue

        for application in prospect_entry.get("prospects_with_details", []):
            applicant_id = application.get("codigo")
            applicant_features = applicants_enhanced.get(applicant_id)

            if not applicant_features:
                continue

            # --- 3. Feature Engineering ---
            skill_match = calculate_skill_match(
                applicant_features.get("technical_skills", []),
                vacancy_features.get("technical_skills", [])
            )
            level_match = calculate_level_match(
                applicant_features.get("experience_level"),
                vacancy_features.get("experience_level")
            )

            # --- 4. Define Target Variable ---
            hired = 1 if application.get("situacao_candidado") == "Contratado pela Decision" else 0

            # --- 5. Assemble the Record ---
            record = {
                "vaga_id": vaga_id,
                "applicant_id": applicant_id,
                "skill_match_score": round(skill_match, 4),
                "level_match_score": level_match,
                # For simplicity, we add the raw features too, which can be useful later
                "applicant_level": applicant_features.get("experience_level"),
                "vacancy_level": vacancy_features.get("experience_level"),
                "applicant_skills_count": len(applicant_features.get("technical_skills", [])),
                "vacancy_skills_count": len(vacancy_features.get("technical_skills", [])),
                # This is our target for the ML model
                "hired": hired
            }
            training_dataset.append(record)

    print(f"-> Created {len(training_dataset)} training records.")

    # --- 6. Save Final Dataset ---
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, indent=4, ensure_ascii=False)
    print(f"-> Saved final training dataset to: {OUTPUT_PATH}")
    print("\n--- Training Dataset Creation Finished Successfully! ---")


if __name__ == "__main__":
    run_dataset_creation()