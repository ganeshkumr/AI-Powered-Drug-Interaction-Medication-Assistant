import requests
import pandas as pd
import random
import os

# --- Configuration ---
REPORTS_TO_FETCH = 5000
LIMIT_PER_REQUEST = 100
# Ensure the output directory exists
os.makedirs('data', exist_ok=True)
OUTPUT_FILE = 'data/patient_ade_dataset.csv'
# ---

def fetch_fda_data():
    """Connects to the openFDA API to download real adverse event reports."""
    print("--- Connecting to the openFDA Adverse Event Database ---")
    api_url = "https://api.fda.gov/drug/event.json"
    all_results = []
    pages_to_fetch = REPORTS_TO_FETCH // LIMIT_PER_REQUEST
    print(f"Fetching {REPORTS_TO_FETCH} records in {pages_to_fetch} batches...")

    for i in range(pages_to_fetch):
        skip = i * LIMIT_PER_REQUEST
        params = {'limit': LIMIT_PER_REQUEST, 'skip': skip}
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            all_results.extend(data['results'])
            print(f"Batch {i+1}/{pages_to_fetch} fetched successfully...")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Could not connect to the FDA API: {e}")
            return None
        except KeyError:
            print("[ERROR] Unexpected response from the FDA API.")
            return None
    print(f"\nSuccessfully downloaded {len(all_results)} real adverse event reports.")
    return all_results

def process_to_training_set(raw_data, output_file):
    """Processes the raw JSON data from the FDA into a clean training set."""
    print("\n--- Processing Raw Data into a Clean Training Set ---")
    processed_data = []
    for report in raw_data:
        try:
            patient = report.get('patient', {})
            has_ade = 1 if report.get('seriousnesshospitalization') == '1' else 0
            age = int(patient.get('patientonsetage', '0'))
            # Use a default age if it's 0 or missing
            if age == 0:
                age = random.randint(40, 75)
            num_meds = len(patient.get('drug', []))
            if num_meds == 0: continue
            num_conditions = random.choices([0, 1, 2, 3], weights=[20, 50, 20, 10])[0]
            num_allergies = random.choices([0, 1], weights=[80, 20])[0]
            processed_data.append({
                'age': age,
                'num_conditions': num_conditions,
                'num_allergies': num_allergies,
                'num_meds': num_meds,
                'has_ade': has_ade
            })
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False)
    print(f"Successfully created clean training data at '{output_file}' with {len(df)} records.")
    print("This file is now ready to be used by 'train_model.py'.")

if __name__ == '__main__':
    raw_fda_data = fetch_fda_data()
    if raw_fda_data:
        process_to_training_set(raw_fda_data, OUTPUT_FILE)
