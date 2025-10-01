import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
# This will load the LLM_API_KEY from your .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# WARNING: Setting this to True will make real API calls, which can be slow and may incur costs.
# It is recommended to test with a small sample size first.
USE_REAL_LLM = False 

# Configure the real Gemini API client
if USE_REAL_LLM:
    try:
        genai.configure(api_key=os.getenv("LLM_API_KEY"))
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        USE_REAL_LLM = False # Fallback to simulation if config fails

def real_gemini_feature_extraction(text: str, entity_type: str) -> dict:
    """
    Makes a real API call to Google's Gemini model to extract structured features.
    
    Args:
        text (str): The input text (CV or job description).
        entity_type (str): 'applicant' or 'vaga' (used in the prompt).

    Returns:
        dict: A dictionary with extracted features.
    """
    if not text:
        return {"technical_skills": [], "languages": {}, "experience_level": "not specified"}

    # Define the Gemini model to use
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Create a detailed prompt for the LLM
    prompt = f"""
    Analyze the following text from a recruitment {entity_type} and extract the information below.
    Return the output ONLY as a valid JSON object.

    1.  "technical_skills": A list of key technical skills, tools, and methodologies. Examples: "Python", "SQL", "AWS", "Scrum", "SAP", "Java".
    2.  "languages": A dictionary of languages and their proficiency levels. Examples: {{"English": "advanced", "Spanish": "intermediate"}}.
    3.  "experience_level": Classify the experience level into one of these four categories ONLY: "junior", "pleno", "senior", or "leadership".

    Here is the text to analyze:
    ---
    {text}
    ---
    """

    try:
        # Make the API call
        response = model.generate_content(prompt)
        # Clean up the response to extract only the JSON part
        json_response = response.text.strip().replace('```json', '').replace('```', '')
        # Parse the JSON string into a Python dictionary
        return json.loads(json_response)
    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")
        # Fallback to the simulation if the real API call fails
        return simulated_gemini_feature_extraction(text, entity_type)

def simulated_gemini_feature_extraction(text: str, entity_type: str) -> dict:
    """
    Simulates a call to an LLM by extracting keywords. This is fast and free.
    """
    if not text:
        return {"technical_skills": [], "languages": {}, "experience_level": "not specified"}
    
    text_lower = text.lower()
    skills_keywords = ['sap', 'python', 'c#', 'java', 'selenium', 'cypress', 'jenkins', 'aws', 'ruby', 'appium', 'cucumber', 'vb.net', 'sql', 'git', 'maven', 'jira', 'scrum', 'kanban', 'power bi', 'docker', 'oracle', '.net', 'react', 'angular', 'peoplesoft', 'abap']
    technical_skills = sorted(list(set([skill for skill in skills_keywords if skill in text_lower])))
    
    languages = {}
    lang_levels = {'english': ['básico', 'intermediário', 'avançado', 'fluente', 'basic', 'intermediate', 'advanced', 'fluent'], 'spanish': ['básico', 'intermediário', 'avançado', 'fluente', 'básico', 'intermedio', 'avanzado', 'fluido']}
    for lang, levels in lang_levels.items():
        pattern = re.compile(f"{lang}[\\s:-]+({'|'.join(levels)})")
        match = pattern.search(text_lower)
        if match:
            languages[lang.capitalize()] = match.group(1)
        elif lang in text_lower:
            languages[lang.capitalize()] = 'not specified'

    experience_level = "not specified"
    if any(keyword in text_lower for keyword in ['leadership', 'gerente', 'coordenador', 'lead']):
        experience_level = 'leadership'
    elif any(keyword in text_lower for keyword in ['senior', 'sênior', 'sr', 'especialista']):
        experience_level = 'senior'
    elif any(keyword in text_lower for keyword in ['pleno', 'pl']):
        experience_level = 'pleno'
    elif any(keyword in text_lower for keyword in ['junior', 'jr']):
        experience_level = 'junior'

    return {"technical_skills": technical_skills, "languages": languages, "experience_level": experience_level}

# --- Main Function ---
# The rest of your application will now call this function.
# It will use the real or simulated version based on the USE_REAL_LLM flag.
def extract_features(text: str, entity_type: str) -> dict:
    if USE_REAL_LLM:
        print("--- Using REAL Gemini API for feature extraction ---")
        return real_gemini_feature_extraction(text, entity_type)
    else:
        # print("--- Using SIMULATED LLM for feature extraction ---")
        return simulated_gemini_feature_extraction(text, entity_type)