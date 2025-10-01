# tests/test_ml.py

import sys
import os

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.create_training_data import calculate_skill_match, calculate_level_match

def run_ml_tests():
    """Executes a series of tests on the ML helper functions and prints the results."""
    print("--- Running ML Pipeline Tests ---")
    
    # Test Suite for calculate_skill_match
    print("\n[TESTING] calculate_skill_match function...")
    test_cases_skill = [
        (calculate_skill_match(['python', 'sql'], ['python', 'sql']), 1.0, "Perfect match"),
        (calculate_skill_match(['python'], ['python', 'sql']), 0.5, "Partial match"),
        (calculate_skill_match(['java'], ['python', 'sql']), 0.0, "No match"),
        (calculate_skill_match([], ['python', 'sql']), 0.0, "No applicant skills"),
        (calculate_skill_match(['python', 'sql'], []), 1.0, "No vacancy skills (should pass)"),
    ]
    
    all_passed = True
    for result, expected, description in test_cases_skill:
        if abs(result - expected) < 1e-9:
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description}: Expected {expected}, Got {result}")
            all_passed = False

    # Test Suite for calculate_level_match
    print("\n[TESTING] calculate_level_match function...")
    test_cases_level = [
        (calculate_level_match('senior', 'senior'), 1.0, "Exact match"),
        (calculate_level_match('leadership', 'senior'), 0.75, "Overqualified"),
        (calculate_level_match('pleno', 'senior'), 0.5, "Slightly underqualified"),
        (calculate_level_match('junior', 'senior'), 0.0, "Too underqualified"),
        (calculate_level_match('not specified', 'senior'), 0.0, "Not specified"),
    ]

    for result, expected, description in test_cases_level:
        if result == expected:
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description}: Expected {expected}, Got {result}")
            all_passed = False
            
    print("\n--- ML Pipeline Tests Complete ---")
    if all_passed:
        print("Result: All tests passed successfully!")
    else:
        print("Result: Some tests failed.")

if __name__ == "__main__":
    run_ml_tests()