import json
import pandas as pd
import os

# --- Configuration: Define all file paths here ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# Input files
PROSPECTS_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'prospects.json')
VAGAS_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'vagas.json')
APPLICANTS_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'applicants.json')
# Output files
MERGED_PROSPECTS_VAGAS_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'prospects_vagas_merged.json')
FILTERED_APPLICANTS_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'applicants_for_processing.json')

def flatten_prospects(prospects_data: dict) -> pd.DataFrame:
    """Flattens the nested prospects.json data into a pandas DataFrame."""
    prospects_list = []
    for vaga_id, vaga_info in prospects_data.items():
        for prospect in vaga_info.get("prospects", []):
            prospect_record = {
                "vaga_id": vaga_id,
                "prospect_nome": prospect.get("nome"),
                "prospect_codigo": prospect.get("codigo"),
                "situacao_candidado": prospect.get("situacao_candidado"),
                "data_candidatura": prospect.get("data_candidatura"),
                "ultima_atualizacao": prospect.get("ultima_atualizacao"),
                "comentario": prospect.get("comentario"),
                "recrutador": prospect.get("recrutador")
            }
            prospects_list.append(prospect_record)
    return pd.DataFrame(prospects_list)


def flatten_vagas(vagas_data: dict) -> pd.DataFrame:
    """Flattens the vagas.json data into a pandas DataFrame."""
    vagas_list = []
    for vaga_id, vaga_details in vagas_data.items():
        record = {
            "vaga_id": vaga_id,
            **vaga_details.get("informacoes_basicas", {}),
            **vaga_details.get("perfil_vaga", {})
        }
        vagas_list.append(record)
    return pd.DataFrame(vagas_list)


def main():
    """
    A unified script to process raw data into two key files:
    1. A merged file of prospects and vacancies.
    2. A filtered list of applicants who are relevant for LLM processing.
    """
    print("--- Starting Unified Data Preparation Pipeline ---")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(MERGED_PROSPECTS_VAGAS_OUTPUT_PATH), exist_ok=True)

    # === Part 1: Merge Prospects and Vacancies ===
    print("\n[Part 1] Merging prospects and vacancies...")
    
    with open(PROSPECTS_INPUT_PATH, 'r', encoding='utf-8') as f:
        prospects_data = json.load(f)
    with open(VAGAS_INPUT_PATH, 'r', encoding='utf-8') as f:
        vagas_data = json.load(f)

    prospects_df = flatten_prospects(prospects_data)
    vagas_df = flatten_vagas(vagas_data)
    
    merged_df = pd.merge(prospects_df, vagas_df, on="vaga_id", how="left")
    
    print(f"-> Saving merged prospects/vacancies data to {MERGED_PROSPECTS_VAGAS_OUTPUT_PATH}")
    merged_df.to_json(MERGED_PROSPECTS_VAGAS_OUTPUT_PATH, orient='records', indent=4, force_ascii=False)
    print("-> Part 1 Complete.")

    # === Part 2: Filter Applicants Based on Merged Data ===
    print("\n[Part 2] Filtering applicants for LLM processing...")

    # Extract unique candidate IDs from the DataFrame we just created
    active_candidate_ids = {str(prospect_codigo) for prospect_codigo in merged_df['prospect_codigo'].unique()}
    print(f"-> Found {len(active_candidate_ids)} unique active candidates.")

    print(f"-> Loading all applicants from {APPLICANTS_INPUT_PATH}...")
    with open(APPLICANTS_INPUT_PATH, 'r', encoding='utf-8') as f:
        all_applicants_data = json.load(f)

    print("-> Filtering applicants...")
    filtered_applicants = {
        applicant_id: applicant_details
        for applicant_id, applicant_details in all_applicants_data.items()
        if applicant_id in active_candidate_ids
    }
    print(f"-> Found {len(filtered_applicants)} matching applicants to process.")

    print(f"-> Saving filtered applicants to {FILTERED_APPLICANTS_OUTPUT_PATH}...")
    with open(FILTERED_APPLICANTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_applicants, f, indent=4, ensure_ascii=False)
    print("-> Part 2 Complete.")

    print("\n--- Unified Data Preparation Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    main()
