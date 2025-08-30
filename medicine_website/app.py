from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import requests
import re # Import the regular expressions library for text parsing

app = Flask(__name__)
app.secret_key = 'your_very_secret_key' 

# --- Helper & API Functions (largely unchanged) ---

def get_db_connection():
    conn = sqlite3.connect('medicine_log.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_rxcui(drug_name):
    base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    params = {"name": drug_name.strip(), "search": 1}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('idGroup', {}).get('rxnormId', [None])[0]
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

def get_interaction_from_api(drug1_name, drug2_name):
    rxcui1 = get_rxcui(drug1_name)
    rxcui2 = get_rxcui(drug2_name)
    if not rxcui1 or not rxcui2: return None
    
    rxcuis_string = f"{rxcui1}+{rxcui2}"
    base_url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"
    params = {"rxcuis": rxcuis_string}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        interaction_groups = data.get('fullInteractionTypeGroup')
        if not interaction_groups or not interaction_groups[0].get('fullInteractionType'):
            return None
        first_interaction = interaction_groups[0]['fullInteractionType'][0]['interactionPair'][0]
        severity = first_interaction.get('severity', 'Unknown')
        description = first_interaction.get('description', 'No description available.')
        if severity.lower() not in ['safe', 'unknown']:
            risk_level = "High Risk" if severity.lower() in ['high', 'n/a'] else "Moderate Risk"
            return f"Interaction between {drug1_name.title()} and {drug2_name.title()} ({risk_level}): {description}"
        return None
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

def check_contraindications(new_drug, patient_conditions, patient_allergies):
    warnings = []
    new_drug_lower = new_drug.lower()
    if patient_allergies and new_drug_lower in patient_allergies.lower():
        warnings.append(f"Allergy Alert: The patient has a known allergy to {new_drug.title()}.")
    contraindication_rules = {'diabetes': ['ibuprofen', 'prednisone'], 'kidney disease': ['ibuprofen', 'naproxen', 'nsaids'], 'high blood pressure': ['ibuprofen', 'pseudoephedrine']}
    if patient_conditions:
        for condition, risky_drugs in contraindication_rules.items():
            if condition in patient_conditions.lower() and new_drug_lower in risky_drugs:
                warnings.append(f"Condition Alert ({condition.title()}): {new_drug.title()} may not be suitable. Consult a doctor.")
    return warnings

# --- NEW: Chatbot Query Processing Logic ---
def process_chatbot_query(question, patient_id):
    """Determines the user's intent and gets the answer."""
    question_lower = question.lower()
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    if not patient:
        return "Please select a valid patient before asking a question."

    # Intent 1: List current medications or dosage
    if "what are my" in question_lower or "list my med" in question_lower or "next dose" in question_lower:
        meds = conn.execute('SELECT drug_name, dosage FROM medications WHERE patient_id = ?', (patient_id,)).fetchall()
        if not meds:
            return f"{patient['name']} currently has no medications logged."
        response = f"Here is the current medication list for {patient['name']}: "
        response += ", ".join([f"{med['drug_name']} ({med['dosage'] or 'dosage not specified'})" for med in meds]) + "."
        if "next dose" in question_lower:
            response += " I can provide the dosage, but I do not have a schedule for when to take them. Please follow your doctor's instructions."
        return response

    # Intent 2: Patient-specific "Can I take [drug]?"
    match = re.search(r'can i take (.+)\?*$', question_lower)
    if match:
        new_drug = match.group(1).strip()
        existing_meds_rows = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (patient_id,)).fetchall()
        existing_meds = [med['drug_name'] for med in existing_meds_rows]
        
        all_warnings = check_contraindications(new_drug, patient['conditions'], patient['allergies'])
        for med in existing_meds:
            interaction = get_interaction_from_api(new_drug, med)
            if interaction:
                all_warnings.append(interaction)
        
        if not all_warnings:
            return f"Based on the available data, no significant interactions or contraindications were found for {new_drug.title()} for {patient['name']}. However, always consult a doctor."
        return "Based on my analysis for " + patient['name'] + ": " + " | ".join(all_warnings)

    # Intent 3: General "What about [drug] and [drug]?"
    match = re.search(r'(what about|interaction between) (.+) and (.+)\?*$', question_lower)
    if match:
        drug1 = match.group(2).strip()
        drug2 = match.group(3).strip()
        interaction = get_interaction_from_api(drug1, drug2)
        return interaction or f"No significant interaction found between {drug1.title()} and {drug2.title()} in the database."

    conn.close()
    return "I'm sorry, I can only answer questions like 'What are my medications?', 'Can I take [drug]?', or 'What about [drug] and [drug]?'"

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    conn = get_db_connection()
    patients_data = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    patients = []
    for p_data in patients_data:
        patient = dict(p_data)
        medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (patient['id'],)).fetchall()
        patient['medications'] = medications
        patients.append(patient)
    conn.close()
    return render_template('index.html', patients=patients)

@app.route('/add_patient', methods=['POST'])
def add_patient():
    name = request.form['name']; age = request.form['age']; conditions = request.form['conditions']; allergies = request.form['allergies']
    conn = get_db_connection()
    conn.execute('INSERT INTO patients (name, age, conditions, allergies) VALUES (?, ?, ?, ?)', (name, age, conditions, allergies))
    conn.commit()
    conn.close()
    flash(f"Patient '{name}' added successfully!", "success")
    return redirect(url_for('index'))

@app.route('/add_medication', methods=['POST'])
def add_medication():
    patient_id = request.form['patient_id']; new_drug = request.form['drug_name']; dosage = request.form['dosage']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    existing_meds_rows = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (patient_id,)).fetchall()
    existing_meds = [med['drug_name'] for med in existing_meds_rows]
    all_warnings = check_contraindications(new_drug, patient['conditions'], patient['allergies'])
    for existing_drug in existing_meds:
        interaction_result = get_interaction_from_api(new_drug, existing_drug)
        if interaction_result:
            all_warnings.append(f"💊 {interaction_result}")
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)', (patient_id, new_drug, dosage))
    conn.commit()
    conn.close()
    flash(f"Medication '{new_drug.title()}' added for {patient['name']}.", "success")
    if all_warnings:
        for warning in all_warnings: flash(warning, "warning")
    else:
        flash("✅ No significant interactions or contraindications found.", "info")
    return redirect(url_for('index'))

@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    """Handles chatbot queries and re-renders the page with the answer."""
    question = request.form.get('question')
    patient_id = request.form.get('patient_id')
    
    chatbot_response = process_chatbot_query(question, patient_id)
    
    # We need to fetch all patient data again to re-render the full page
    conn = get_db_connection()
    patients_data = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    patients = []
    for p_data in patients_data:
        patient = dict(p_data)
        medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (patient['id'],)).fetchall()
        patient['medications'] = medications
        patients.append(patient)
    conn.close()
    
    return render_template('index.html', patients=patients, chatbot_response=chatbot_response, selected_patient_id=int(patient_id))

if __name__ == '__main__':
    app.run(debug=True)

