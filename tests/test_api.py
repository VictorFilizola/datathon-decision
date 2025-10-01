# tests/test_api.py

import requests
import subprocess
import time
import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"

def run_api_tests():
    """
    Starts the API server, runs a series of requests to test each endpoint,
    and then shuts the server down.
    """
    server_process = None
    all_passed = True
    
    try:
        # 1. Start the FastAPI server as a background process
        print("--- Starting API Server for Testing ---")
        # Note: We assume the command is run from the project root.
        # Using sys.executable ensures we use the same Python interpreter.
        server_command = [
            sys.executable, "-m", "uvicorn", 
            "backend.main:app", "--host", "127.0.0.1", "--port", "8000"
        ]
        server_process = subprocess.Popen(server_command)
        print(f"Server process started with PID: {server_process.pid}. Waiting for it to initialize...")
        time.sleep(5)  # Wait for the server to be ready

        # --- Test Suite ---
        print("\n--- Running API Endpoint Tests ---")

        # Test 1: Root endpoint
        print("\n[TESTING] GET /")
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                print(f"  [PASS] Status code is 200. Response: {response.json()}")
            else:
                print(f"  [FAIL] Expected status 200, Got {response.status_code}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"  [FAIL] Request failed: {e}")
            all_passed = False

        # Test 2: List models endpoint
        print("\n[TESTING] GET /models")
        try:
            response = requests.get(f"{API_URL}/models")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"  [PASS] Status code is 200. Found {len(models)} model(s).")
                # Store the latest model for subsequent tests
                latest_model = models[0] if models else None
            else:
                print(f"  [FAIL] Expected status 200, Got {response.status_code}")
                latest_model = None
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"  [FAIL] Request failed: {e}")
            latest_model = None
            all_passed = False

        # Test 3: Evaluate model endpoint
        if latest_model:
            print(f"\n[TESTING] GET /evaluate/{latest_model}")
            try:
                response = requests.get(f"{API_URL}/evaluate/{latest_model}")
                if response.status_code == 200 and "accuracy" in response.json():
                    print(f"  [PASS] Status code is 200. Model accuracy: {response.json()['accuracy']}")
                else:
                    print(f"  [FAIL] Expected status 200 and accuracy key, Got {response.status_code}")
                    all_passed = False
            except requests.exceptions.RequestException as e:
                print(f"  [FAIL] Request failed: {e}")
                all_passed = False
        else:
            print("\n[SKIPPED] GET /evaluate - No model found to evaluate.")
            
        # Test 4: Predict endpoint
        if latest_model:
            print(f"\n[TESTING] POST /predict/{latest_model}")
            # Use a sample applicant JSON for the test payload
            applicant_payload = {
                "codigo_profissional": "test-999",
                "cv_pt": "Desenvolvedor de software com grande experiência em Python e SQL para análise de dados e machine learning. Inglês avançado."
            }
            try:
                response = requests.post(f"{API_URL}/predict/{latest_model}", json=applicant_payload)
                if response.status_code == 200 and "top_matches" in response.json():
                    print(f"  [PASS] Status code is 200. Prediction successful.")
                    print(f"  Top match Vaga ID: {response.json()['top_matches'][0]['vaga_id']}")
                else:
                    print(f"  [FAIL] Expected status 200 and top_matches key, Got {response.status_code}")
                    all_passed = False
            except requests.exceptions.RequestException as e:
                print(f"  [FAIL] Request failed: {e}")
                all_passed = False
        else:
            print("\n[SKIPPED] POST /predict - No model found to use for prediction.")

    finally:
        # 4. Shut down the server process
        if server_process:
            print("\n--- Shutting Down API Server ---")
            server_process.terminate()
            server_process.wait()
            print("Server process terminated.")
            
    print("\n--- API Tests Complete ---")
    if all_passed:
        print("Result: All tests passed successfully!")
    else:
        print("Result: Some tests failed.")


if __name__ == "__main__":
    run_api_tests()