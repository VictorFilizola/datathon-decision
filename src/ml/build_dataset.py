import json
import os
import random

# --- Configuration ---
# Define paths using the recommended project structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROSPECTS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'prospects.json')
VAGAS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'vagas.json')
APPLICANTS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'applicants.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'prospects_aggregated.json')
SAMPLE_SIZE = 5000

def run_aggregation():
    """
    Main function to load data, perform the aggregation, and save the result.
    """
    print("--- Starting Data Aggregation Pipeline ---")

    # --- 1. Load Data ---
    print(f"-> Loading prospects from: {PROSPECTS_PATH}")
    with open(PROSPECTS_PATH, 'r', encoding='utf-8') as f:
        prospects_data = json.load(f)

    print(f"-> Loading vacancies from: {VAGAS_PATH}")
    with open(VAGAS_PATH, 'r', encoding='utf-8') as f:
        vagas_data = json.load(f)

    print(f"-> Loading applicants from: {APPLICANTS_PATH}")
    with open(APPLICANTS_PATH, 'r', encoding='utf-8') as f:
        applicants_data = json.load(f)
    print("-> All data loaded successfully.")

    # --- 2. Sample Prospects ---
    prospect_keys = list(prospects_data.keys())
    if len(prospect_keys) < SAMPLE_SIZE:
        print(f"Warning: Total prospects ({len(prospect_keys)}) is less than sample size ({SAMPLE_SIZE}). Using all prospects.")
        sampled_keys = prospect_keys
    else:
        sampled_keys = random.sample(prospect_keys, SAMPLE_SIZE)
    print(f"\n-> Randomly selected {len(sampled_keys)} prospects for aggregation.")

    # --- 3. Aggregate Data ---
    print("-> Aggregating vacancy and applicant details...")
    aggregated_data = []
    processed_count = 0
    for key in sampled_keys:
        prospect_info = prospects_data.get(key)
        vaga_info = vagas_data.get(key)

        # Skip if the prospect ID doesn't have a matching vacancy
        if not prospect_info or not vaga_info:
            continue

        aggregated_prospect = {
            "vaga_id": key,
            "vaga_details": vaga_info,
            "prospects_with_details": []
        }

        # Iterate through each applicant for the current prospect
        for applicant_summary in prospect_info.get("prospects", []):
            applicant_id = applicant_summary.get("codigo")
            # Fetch the full applicant profile
            if applicant_id and applicant_id in applicants_data:
                full_applicant_details = applicants_data[applicant_id]
                # Combine the summary (like 'situacao_candidado') with the full profile
                combined_applicant_data = {
                    **applicant_summary,
                    "full_profile": full_applicant_details
                }
                aggregated_prospect["prospects_with_details"].append(combined_applicant_data)

        # Only add the record if it contains applicants with full profiles
        if aggregated_prospect["prospects_with_details"]:
            aggregated_data.append(aggregated_prospect)
            processed_count += 1

    print(f"-> Successfully aggregated {processed_count} prospects with their full data.")

    # --- 4. Save Output ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\n-> Saving aggregated file to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=4, ensure_ascii=False)

    print("\n--- Data Aggregation Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    run_aggregation()