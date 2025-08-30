from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import requests # Import the requests library for API calls

app = Flask(__name__)
# Flask needs a secret key to display flashed messages
app.secret_key = 'your_very_secret_key' 

# --- Helper Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('medicine_log.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- API Interaction Logic (from our previous chatbot) ---

def get_rxcui(drug_name):
    """Gets the RxCUI identifier for a given drug name from the NIH API."""
    base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    params = {"name": drug_name.strip(), "search": 1}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        rxcui = data.get('idGroup', {}).get('rxnormId', [None])[0]
        return rxcui
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

def get_interaction_from_api(drug1_name, drug2_name):
    """Finds drug-drug interactions using the NIH API."""
    rxcui1 = get_rxcui(drug1_name)
    rxcui2 = get_rxcui(drug2_name)

    if not rxcui1 or not rxcui2:
        return None # Return None if a drug can't be identified

    rxcuis_string = f"{rxcui1}+{rxcui2}"
    base_url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"
    params = {"rxcuis": rxcuis_string}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        interaction_groups = data.get('fullInteractionTypeGroup')
        if not interaction_groups or not interaction_groups[0].get('fullInteractionType'):
            return None # No interaction found

        first_interaction = interaction_groups[0]['fullInteractionType'][0]['interactionPair'][0]
        severity = first_interaction.get('severity', 'Unknown')
        description = first_interaction.get('description', 'No description available.')
        
        if severity.lower() != 'safe' and severity.lower() != 'unknown':
            risk_level = "High Risk" if severity.lower() in ['high', 'n/a'] else "Moderate Risk"
            return f"{risk_level} with {drug2_name.title()}: {description}"
        return None
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

# --- Contraindication & Allergy Check Logic ---
def check_contraindications(new_drug, patient_conditions, patient_allergies):
    """
    A simple rule-based check for contraindications and allergies.
    This can be expanded with more rules.
    """
    warnings = []
    new_drug_lower = new_drug.lower()
    
    # 1. Check for Allergies
    if patient_allergies and new_drug_lower in patient_allergies.lower():
        warnings.append(f"🚨 Allergy Alert: The patient has a known allergy to {new_drug.title()}.")
        
    # 2. Check for Health Conditions (simple example rules)
    contraindication_rules = {
        'diabetes': ['ibuprofen', 'prednisone'],
        'kidney disease': ['ibuprofen', 'naproxen', 'nsaids'],
        'high blood pressure': ['ibuprofen', 'pseudoephedrine']
    }
    
    if patient_conditions:
        for condition, risky_drugs in contraindication_rules.items():
            if condition in patient_conditions.lower() and new_drug_lower in risky_drugs:
                warnings.append(f"⚠️ Condition Alert ({condition.title()}): {new_drug.title()} may not be suitable. Consult a doctor.")
                
    return warnings

# --- Flask Routes ---

@app.route('/')
def index():
    """Main route to display all patient and medication data."""
    conn = get_db_connection()
    patients_data = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    patients = []
    for p_data in patients_data:
        patient = dict(p_data)
        medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (patient['id'],)).fetchall()
        patient['medications'] = medications
        patients.append(patient)
    conn.close()
    return render_template('index.html', patients=patients)

@app.route('/add_patient', methods=['POST'])
def add_patient():
    """Route to handle adding a new patient."""
    name = request.form['name']
    age = request.form['age']
    conditions = request.form['conditions']
    allergies = request.form['allergies']

    conn = get_db_connection()
    conn.execute('INSERT INTO patients (name, age, conditions, allergies) VALUES (?, ?, ?, ?)',
                 (name, age, conditions, allergies))
    conn.commit()
    conn.close()
    flash(f"Patient '{name}' added successfully!", "success")
    return redirect(url_for('index'))

@app.route('/add_medication', methods=['POST'])
def add_medication():
    """Route to add a new medication and perform all interaction checks."""
    patient_id = request.form['patient_id']
    new_drug = request.form['drug_name']
    dosage = request.form['dosage']

    conn = get_db_connection()
    
    # --- 1. Get Patient's Existing Data ---
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    existing_meds_rows = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (patient_id,)).fetchall()
    existing_meds = [med['drug_name'] for med in existing_meds_rows]

    # --- 2. Perform All Checks ---
    # a) Check contraindications and allergies
    all_warnings = check_contraindications(new_drug, patient['conditions'], patient['allergies'])
    
    # b) Check for drug-drug interactions against existing meds
    for existing_drug in existing_meds:
        interaction_result = get_interaction_from_api(new_drug, existing_drug)
        if interaction_result:
            all_warnings.append(f"💊 Interaction Alert: {interaction_result}")

    # --- 3. Save the New Medication to the Database ---
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)',
                 (patient_id, new_drug, dosage))
    conn.commit()
    conn.close()

    # --- 4. Flash Messages to the User ---
    flash(f"Medication '{new_drug.title()}' added for {patient['name']}.", "success")
    if all_warnings:
        for warning in all_warnings:
            flash(warning, "warning") # Use "warning" category for alerts
    else:
        flash("✅ No significant interactions or contraindications found.", "info")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

