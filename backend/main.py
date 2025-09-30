import os
import glob
import joblib
import pandas as pd
import uvicorn
import json
import logging  # Importa o módulo de logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Configuração do Logging ---
# Cria um logger específico para as predições
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.setLevel(logging.INFO)
# Define um handler para salvar os logs em um arquivo
# Usamos 'a' para modo append, para não apagar os logs anteriores
handler = logging.FileHandler("predictions.log", mode='a')
# Define o formato do log
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
prediction_logger.addHandler(handler)

# --- Configuração de Caminhos e Dados ---
# (O restante da configuração permanece o mesmo)
# --- Project Structure Setup ---
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.ml.feature_extractor import simulated_gemini_feature_extraction
from src.ml.create_training_data import calculate_skill_match, calculate_level_match
from src.ml.train import run_training_pipeline as trigger_training

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Recruitment Matching API",
    description="An API to train, evaluate, and use a model for matching candidates to vacancies.",
    version="1.5.0" # Version bump for monitoring
)

# --- Configuration & Data Loading ---
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

VACANCIES_ENHANCED_PATH = os.path.join(PROCESSED_DATA_DIR, 'vacancies_enhanced.json')
TRAINING_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'training_dataset.json')
VAGAS_RAW_PATH = os.path.join(RAW_DATA_DIR, 'vagas.json')

try:
    with open(VAGAS_RAW_PATH, 'r', encoding='utf-8') as f:
        VAGAS_RAW_DATA = json.load(f)
    with open(VACANCIES_ENHANCED_PATH, 'r', encoding='utf-8') as f:
        VACANCIES_ENHANCED_DATA = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading data on startup: {e}. Ensure all data files are present.")
    VAGAS_RAW_DATA = {}
    VACANCIES_ENHANCED_DATA = {}


# --- Pydantic Models ---
class RawApplicant(BaseModel):
    cv_pt: str = Field(..., description="The full CV text of the applicant in Portuguese.")
    codigo_profissional: Optional[str] = Field(None, description="The applicant's professional ID code.")
    class Config:
        extra = 'allow'

# --- Helper Functions ---
def get_latest_model_path():
    list_of_models = glob.glob(os.path.join(MODEL_DIR, '*.joblib'))
    if not list_of_models:
        return None
    return max(list_of_models, key=os.path.getctime)

# --- API Routes ---
# (As rotas /train, /models, /evaluate permanecem as mesmas)
@app.get("/")
def index():
    return {"message": "Recruitment Model API is running."}

@app.post("/train", status_code=201)
def train_model_endpoint():
    try:
        trigger_training()
        latest_model = get_latest_model_path()
        return {"status": "success", "message": "Model training complete.", "new_model_file": os.path.basename(latest_model) if latest_model else "N/A"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    models = [os.path.basename(f) for f in glob.glob(os.path.join(MODEL_DIR, '*.joblib'))]
    if not models:
        raise HTTPException(status_code=404, detail="No models found.")
    return {"status": "success", "models": sorted(models, reverse=True)}

@app.get("/evaluate/{model_filename}")
def evaluate_specific_model(model_filename: str):
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found.")
    try:
        model = joblib.load(model_path)
        df = pd.read_json(TRAINING_DATASET_PATH)
        features = ['skill_match_score', 'level_match_score', 'applicant_skills_count', 'vacancy_skills_count']
        target = 'hired'
        _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42, stratify=df[target])
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, target_names=['Not Hired', 'Hired'], output_dict=True, zero_division=0)
        return {"model_filename": model_filename, "accuracy": f"{accuracy:.2%}", "confusion_matrix": conf_matrix.tolist(), "classification_report": class_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_filename}")
def predict_match(model_filename: str, applicant_raw: RawApplicant):
    try:
        # (Lógica de carregamento do modelo permanece a mesma)
        if model_filename == "latest":
            model_path = get_latest_model_path()
            if not model_path:
                raise HTTPException(status_code=500, detail="No trained model found.")
        else:
            model_path = os.path.join(MODEL_DIR, model_filename)
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found.")
        
        model = joblib.load(model_path)
        
        applicant_features = simulated_gemini_feature_extraction(applicant_raw.cv_pt, 'applicant')

        prediction_records = []
        for vaga_id, vacancy_features in VACANCIES_ENHANCED_DATA.items():
            # (Lógica de feature engineering permanece a mesma)
            skill_match = calculate_skill_match(applicant_features.get("technical_skills", []), vacancy_features.get("technical_skills", []))
            level_match = calculate_level_match(applicant_features.get("experience_level"), vacancy_features.get("experience_level"))
            record = {"vaga_id": vaga_id, "skill_match_score": skill_match, "level_match_score": level_match, "applicant_skills_count": len(applicant_features.get("technical_skills", [])), "vacancy_skills_count": len(vacancy_features.get("technical_skills", []))}
            prediction_records.append(record)

        df_predict = pd.DataFrame(prediction_records)
        feature_order = ['skill_match_score', 'level_match_score', 'applicant_skills_count', 'vacancy_skills_count']
        probabilities = model.predict_proba(df_predict[feature_order])[:, 1]
        df_predict['match_probability'] = probabilities
        
        # --- NOVO: Logging da Predição ---
        # Para cada predição, salvamos as features e a probabilidade em nosso arquivo de log.
        for record in df_predict.to_dict(orient='records'):
            prediction_logger.info(json.dumps(record))
        # --- FIM DO LOGGING ---

        top_5_matches_df = df_predict.sort_values(by='match_probability', ascending=False).head(5)
        
        # (Lógica de enriquecimento da resposta permanece a mesma)
        top_matches_enriched = []
        for match in top_5_matches_df.to_dict(orient='records'):
            vaga_id = match['vaga_id']
            raw_vaga = VAGAS_RAW_DATA.get(vaga_id, {})
            basic_info = raw_vaga.get("informacoes_basicas", {})
            profile_info = raw_vaga.get("perfil_vaga", {})
            
            match['vaga_details'] = {
                "title": basic_info.get("titulo_vaga", "N/A"),
                "client": basic_info.get("cliente", "N/A"),
                "contract_type": basic_info.get("tipo_contratacao", "N/A"),
                "main_activities": profile_info.get("principais_atividades", "N/A")
            }
            top_matches_enriched.append(match)

        return {
            "status": "success",
            "applicant_id": applicant_raw.codigo_profissional or "N/A",
            "model_used": os.path.basename(model_path),
            "applicant_extracted_features": applicant_features,
            "top_matches": top_matches_enriched
        }
    except Exception as e:
        # Adiciona log de erro também
        logging.error(f"Prediction failed with error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)