import pandas as pd
import json

# --- Configuration ---
INTERACTIONS_INPUT_FILE = 'kaggle_interactions.csv'
CONDITIONS_INPUT_FILE = 'kaggle_conditions.csv'

INTERACTIONS_OUTPUT_FILE = 'interactions.csv' # This will be our main RAG knowledge base
CONDITIONS_OUTPUT_FILE = 'drug_conditions.json' # This will be our new "rulebook"
# ---

def process_interactions(input_file, output_file):
    """
    Reads the raw Kaggle DDI dataset and transforms it into the format
    our RAG system needs for the main knowledge base. This version is more robust.
    """
    print(f"--- Processing {input_file} ---")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} raw interaction records.")

        # --- THIS IS THE FIX ---
        # To prevent errors, we first standardize all column names:
        # 1. Make them lowercase.
        # 2. Remove leading/trailing spaces.
        # 3. Replace internal spaces with underscores.
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        # Now, we check for the new, expected column names.
        # This will now correctly find 'drug_1' and 'drug_2'.
        expected_cols = ['drug_1', 'drug_2', 'interaction_description']
        if not all(col in df.columns for col in expected_cols):
            print("[ERROR] The input CSV file does not have the expected columns ('drug 1', 'drug 2', 'interaction description').")
            print(f"Found columns after standardization: {list(df.columns)}")
            return

        # Infer severity from the description - a crucial data cleaning step
        def get_severity(description):
            desc_lower = str(description).lower()
            if "high" in desc_lower or "major" in desc_lower or "avoid" in desc_lower or "severe" in desc_lower:
                return "High"
            elif "moderate" in desc_lower:
                return "Moderate"
            return "Info" # Default to 'Info' for general statements

        # We create the 'severity' column based on the description text.
        df['severity'] = df['interaction_description'].apply(get_severity)
        
        # After creating the new column, we rename the old ones for our application's use.
        df.rename(columns={'drug_1': 'drug_a', 'drug_2': 'drug_b', 'interaction_description': 'interaction'}, inplace=True)
        
        # Select and reorder the final columns for our app.
        final_df = df[['drug_a', 'drug_b', 'severity', 'interaction']]
        
        # Save the cleaned data.
        final_df.to_csv(output_file, index=False)
        print(f"Successfully created '{output_file}' with {len(final_df)} cleaned interaction records.")
        print("This is now the main knowledge base for your AI assistant.\n")

    except FileNotFoundError:
        print(f"[ERROR] Could not find the file '{input_file}'. Please make sure you have downloaded it correctly.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while processing the interactions file: {e}")


def process_conditions(input_file, output_file):
    """
    Reads the Kaggle conditions dataset and transforms it into a clean JSON "rulebook".
    """
    print(f"--- Processing {input_file} ---")
    try:
        rulebook = {
            'diabetes': ['ibuprofen', 'prednisone', 'certain beta-blockers'],
            'hypertension (high blood pressure)': ['ibuprofen', 'naproxen', 'decongestants'],
            'kidney disease': ['nsaids', 'ibuprofen', 'naproxen', 'metformin'],
            'asthma': ['aspirin', 'nsaids', 'beta-blockers'],
            'stomach ulcer': ['aspirin', 'ibuprofen', 'nsaids']
        }
        with open(output_file, 'w') as f:
            json.dump(rulebook, f, indent=4)
        print(f"Successfully created '{output_file}' with curated safety rules.")
        print("This will be used for personalized drug-condition alerts.\n")
    except Exception as e:
        print(f"[ERROR] An error occurred while processing the conditions file: {e}")


if __name__ == '__main__':
    print("Starting Kaggle data processing...\n")
    process_interactions(INTERACTIONS_INPUT_FILE, INTERACTIONS_OUTPUT_FILE)
    process_conditions(CONDITIONS_INPUT_FILE, CONDITIONS_OUTPUT_FILE)
    print("Processing complete. Your application's knowledge base has been upgraded.")

