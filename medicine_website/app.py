from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import requests

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_a_great_project' 

# --- Helper Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('medicine_log.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- API & Safety Check Logic ---
def get_rxcui(drug_name):
    """Gets the RxCUI identifier for a drug from the NIH API."""
    base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    params = {"name": drug_name.strip(), "search": 1}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('idGroup', {}).get('rxnormId', [None])[0]
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

def get_interaction(drug1_name, drug2_name):
    """Checks for interactions between two drugs."""
    rxcui1 = get_rxcui(drug1_name); rxcui2 = get_rxcui(drug2_name)
    if not rxcui1 or not rxcui2: return None
    
    rxcuis_string = f"{rxcui1}+{rxcui2}"
    base_url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"
    params = {"rxcuis": rxcuis_string}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        group = data.get('fullInteractionTypeGroup')
        if not group or not group[0].get('fullInteractionType'): return None
        
        interaction = group[0]['fullInteractionType'][0]['interactionPair'][0]
        severity = interaction.get('severity', 'Unknown')
        description = interaction.get('description', 'N/A')
        
        if severity.lower() not in ['safe', 'unknown']:
            risk = "High Risk" if severity.lower() in ['high', 'n/a'] else "Moderate Risk"
            return f"💊 Drug-Drug Interaction ({risk}): {description}"
        return None
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

def get_personalized_warnings(new_drug, patient):
    """Generates warnings based on the patient's specific profile."""
    warnings = []
    drug_lower = new_drug.lower()
    conditions = patient['conditions'].lower() if patient['conditions'] else ""
    allergies = patient['allergies'].lower() if patient['allergies'] else ""

    # 1. Allergy Check
    if allergies and any(allergy.strip() in drug_lower for allergy in allergies.split(',')):
        warnings.append(f"🚨 Allergy Alert: Profile indicates an allergy to '{new_drug}'. Do not take this medication.")

    # 2. Drug-Condition Check (Contraindications)
    rules = {
        'ibuprofen': ['diabetes', 'kidney disease', 'high blood pressure'],
        'metformin': ['kidney disease'],
        'aspirin': ['stomach ulcers', 'bleeding disorder']
    }
    if drug_lower in rules:
        for condition in rules[drug_lower]:
            if condition in conditions:
                warnings.append(f"⚠️ Drug-Condition Alert: Taking '{new_drug.title()}' with a history of '{condition.title()}' can be risky. Consult your doctor.")
    
    return warnings

# --- USER & PROFILE MANAGEMENT ---
@app.route('/', methods=['GET', 'POST'])
def login():
    """Handles user login and registration."""
    session.pop('patient_id', None) # Clear any previous session
    if request.method == 'POST':
        patient_name = request.form['name'].strip()
        conn = get_db_connection()
        patient = conn.execute('SELECT * FROM patients WHERE name = ?', (patient_name,)).fetchone()
        
        if patient:
            session['patient_id'] = patient['id']
            return redirect(url_for('dashboard'))
        else: # If patient doesn't exist, create them
            cursor = conn.cursor()
            cursor.execute('INSERT INTO patients (name) VALUES (?)', (patient_name,))
            conn.commit()
            new_patient = conn.execute('SELECT * FROM patients WHERE name = ?', (patient_name,)).fetchone()
            session['patient_id'] = new_patient['id']
            flash("Welcome! Let's set up your health profile.", "info")
            return redirect(url_for('profile'))
    return render_template('index.html', page='login')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """The main view for a logged-in patient."""
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    
    return render_template('index.html', page='dashboard', patient=patient, medications=medications)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """Handles creating and updating a patient's health profile."""
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    conn = get_db_connection()
    if request.method == 'POST':
        conn.execute('''
            UPDATE patients SET age = ?, gender = ?, weight_kg = ?, conditions = ?, allergies = ?
            WHERE id = ?
        ''', (request.form['age'], request.form['gender'], request.form['weight_kg'], 
              request.form['conditions'], request.form['allergies'], session['patient_id']))
        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('dashboard'))
        
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    return render_template('index.html', page='profile', patient=patient)

# --- MEDICATION MANAGEMENT ---
@app.route('/add_medication', methods=['POST'])
def add_medication():
    """Adds a new medication and performs all safety checks."""
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    new_drug = request.form['drug_name']
    dosage = request.form['dosage']
    
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    
    all_warnings = get_personalized_warnings(new_drug, patient)
    
    existing_meds = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    for med in existing_meds:
        interaction = get_interaction(new_drug, med['drug_name'])
        if interaction:
            all_warnings.append(interaction)
    
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)',
                 (session['patient_id'], new_drug, dosage))
    conn.commit()
    conn.close()
    
    flash(f"'{new_drug.title()}' has been added to your log.", "success")
    if all_warnings:
        for warning in all_warnings:
            flash(warning, "warning")
    else:
        flash("✅ No personalized conflicts or major interactions were found.", "info")
        
    return redirect(url_for('dashboard'))

@app.route('/logout', methods=['GET'])
def logout():
    """Logs the user out by clearing the session."""
    session.pop('patient_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
