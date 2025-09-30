import json
import os
import re
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file at the project root
# This will load your LLM_API_KEY
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'prospects_aggregated.json')
OUTPUT_VACANCIES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'vacancies_enhanced.json')
OUTPUT_APPLICANTS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'applicants_enhanced.json')

# --- Simulated Gemini API Call ---
# In a real-world scenario, this function would make an API call to Gemini.
# For this project, we simulate its behavior by extracting keywords.
def simulated_gemini_feature_extraction(text, entity_type):
    """
    Simulates a call to an LLM to extract structured features from text.

    Args:
        text (str): The input text (CV or job description).
        entity_type (str): 'applicant' or 'vaga'.

    Returns:
        dict: A dictionary with extracted features.
    """
    if not text:
        return {
            "technical_skills": [],
            "languages": {},
            "experience_level": "not specified"
        }

    # Normalize text to lowercase for consistent searching
    text_lower = text.lower()

    # 1. Technical Skills Extraction
    skills_keywords = [
        'sap', 'python', 'web engineering', 'c#', 'java', 'selenium', 'cypress', 'jenkins',
        'aws', 'ruby', 'appium', 'cucumber', 'vb.net', 'sql', 'git', 'maven', 'jira', 'scrum',
        'kanban', 'metodologia ágil', 'power bi', 'docker', 'oracle', '.net', 'react', 'angular',
        'peoplesoft', 'control m', 'pega', 'salesforce', 'siebel', 'abap', 'cobol'
    ]
    technical_skills = sorted(list(set([skill for skill in skills_keywords if skill in text_lower])))

    # 2. Language Extraction
    languages = {}
    lang_levels = {
        'english': ['básico', 'intermediário', 'avançado', 'fluente'],
        'spanish': ['básico', 'intermediário', 'avançado', 'fluente']
    }
    # Using a more robust regex to find language and its level
    for lang, levels in lang_levels.items():
        # Search for "ingles/espanhol" followed by "level"
        pattern = re.compile(f"{lang}[\\s:-]+({'|'.join(levels)})")
        match = pattern.search(text_lower)
        if match:
            languages[lang.capitalize()] = match.group(1)
        # If level is not specified but the language is mentioned
        elif lang in text_lower:
            languages[lang.capitalize()] = 'not specified'


    # 3. Experience Level Extraction
    experience_level = "not specified"
    if any(keyword in text_lower for keyword in ['leadership', 'gerente', 'coordenador', 'lead']):
        experience_level = 'leadership'
    elif any(keyword in text_lower for keyword in ['senior', 'sênior', 'sr', 'especialista']):
        experience_level = 'senior'
    elif any(keyword in text_lower for keyword in ['pleno', 'pl']):
        experience_level = 'pleno'
    elif any(keyword in text_lower for keyword in ['junior', 'jr']):
        experience_level = 'junior'

    return {
        "technical_skills": technical_skills,
        "languages": languages,
        "experience_level": experience_level
    }

def run_feature_extraction():
    """
    Main function to load aggregated data, extract features using the LLM,
    and save the enhanced data.
    """
    print("--- Starting LLM Feature Extraction Pipeline ---")

    # --- 1. Load Aggregated Data ---
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            aggregated_data = json.load(f)
        print(f"-> Successfully loaded {len(aggregated_data)} prospects from {INPUT_PATH}")
    except FileNotFoundError:
        print(f"Error: The file {INPUT_PATH} was not found.")
        print("Please run the 'build_dataset.py' script first to generate it.")
        return

    enhanced_vacancies = {}
    enhanced_applicants = {}

    print("-> Processing vacancies and applicants...")
    # --- 2. Iterate and Extract Features ---
    for prospect in aggregated_data:
        vaga_id = prospect.get("vaga_id")
        vaga_details = prospect.get("vaga_details")

        # Process Vacancy if not already processed
        if vaga_id and vaga_details and vaga_id not in enhanced_vacancies:
            description = (
                vaga_details.get("informacoes_basicas", {}).get("titulo_vaga", "") + " " +
                vaga_details.get("perfil_vaga", {}).get("principais_atividades", "") + " " +
                vaga_details.get("perfil_vaga", {}).get("competencia_tecnicas_e_comportamentais", "")
            )
            features = simulated_gemini_feature_extraction(description, 'vaga')
            enhanced_vacancies[vaga_id] = {
                "vaga_id": vaga_id,
                **features
            }

        # Process Applicants if not already processed
        for applicant in prospect.get("prospects_with_details", []):
            applicant_id = applicant.get("codigo")
            if applicant_id and applicant_id not in enhanced_applicants:
                cv_text = applicant.get("full_profile", {}).get("cv_pt", "")
                features = simulated_gemini_feature_extraction(cv_text, 'applicant')
                enhanced_applicants[applicant_id] = {
                    "applicant_id": applicant_id,
                    **features
                }

    print(f"-> Extracted features for {len(enhanced_vacancies)} unique vacancies.")
    print(f"-> Extracted features for {len(enhanced_applicants)} unique applicants.")

    # --- 3. Save Enhanced Data ---
    os.makedirs(os.path.dirname(OUTPUT_VACANCIES_PATH), exist_ok=True)
    with open(OUTPUT_VACANCIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(enhanced_vacancies, f, indent=4, ensure_ascii=False)
    print(f"-> Saved enhanced vacancy data to: {OUTPUT_VACANCIES_PATH}")

    with open(OUTPUT_APPLICANTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(enhanced_applicants, f, indent=4, ensure_ascii=False)
    print(f"-> Saved enhanced applicant data to: {OUTPUT_APPLICANTS_PATH}")

    print("\n--- LLM Feature Extraction Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    # Example of how to retrieve the API key from environment variables
    llm_api_key = os.getenv("LLM_API_KEY")
    if not llm_api_key or llm_api_key == "your_super_secret_api_key_here":
        print("Warning: LLM_API_KEY is not set in your .env file. The script will run with the simulated API.")
    else:
        print("LLM_API_KEY loaded successfully.")

    run_feature_extraction()