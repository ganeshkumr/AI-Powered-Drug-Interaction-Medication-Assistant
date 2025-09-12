import pandas as pd
import random

# --- Configuration ---
NUM_PATIENTS = 5000
CONDITIONS_LIST = ['Diabetes', 'Hypertension', 'Kidney Disease', 'Liver Disease', 'Asthma', 'Heart Disease']
ALLERGENS_LIST = ['Penicillin', 'Sulfa', 'Aspirin', 'Ibuprofen', 'Codeine']

# --- Data Generation ---
data = []
for i in range(NUM_PATIENTS):
    age = random.randint(20, 95)
    num_conditions = random.choices([0, 1, 2, 3, 4], weights=[20, 40, 25, 10, 5])[0]
    num_allergies = random.choices([0, 1, 2], weights=[70, 25, 5])[0]
    num_meds = random.randint(1, 12)
    
    # --- Risk Logic for ADE ---
    # The core of the simulation. We define what factors increase the risk of an ADE.
    base_risk = 0.05  # 5% baseline risk for anyone
    
    # Risk increases with age
    if age > 65: base_risk += 0.15
    if age > 80: base_risk += 0.10
        
    # Risk increases with number of medications (polypharmacy)
    if num_meds > 5: base_risk += 0.20
    if num_meds > 8: base_risk += 0.15
        
    # Risk increases with number of conditions and allergies
    base_risk += num_conditions * 0.08
    base_risk += num_allergies * 0.05
    
    # Determine if an ADE occurred based on the calculated risk
    has_ade = 1 if random.random() < base_risk else 0
    
    data.append({
        'age': age,
        'num_conditions': num_conditions,
        'num_allergies': num_allergies,
        'num_meds': num_meds,
        'has_ade': has_ade # This is our target variable: 1 for yes, 0 for no
    })

# --- Create and Save DataFrame ---
df = pd.DataFrame(data)
df.to_csv('patient_ade_dataset.csv', index=False)

print(f"Successfully generated a synthetic dataset with {NUM_PATIENTS} patient records.")
print(f"Dataset saved to 'patient_ade_dataset.csv'.")
print("\nSample of the generated data:")
print(df.head())
print(f"\nTotal simulated ADEs in dataset: {df['has_ade'].sum()}")
