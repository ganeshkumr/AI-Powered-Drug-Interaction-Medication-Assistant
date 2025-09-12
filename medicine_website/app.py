from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date
import joblib # To load our trained ML model
import numpy as np

app = Flask(__name__)
app.secret_key = 'the_final_and_most_secure_key' 

# --- Load the Prediction Model ---
try:
    ade_predictor = joblib.load('ade_predictor.joblib')
    print("[INFO] Adverse Drug Event (ADE) prediction model loaded successfully.")
except FileNotFoundError:
    print("[ERROR] 'ade_predictor.joblib' not found. Please run train_model.py first.")
    ade_predictor = None

# --- RAG System (Unchanged) ---
class RAGSystem:
    def __init__(self, data_file):
        try:
            self.df = pd.read_csv(data_file)
            self.df['drug_a_lower'] = self.df['drug_a'].astype(str).str.lower().str.strip()
            self.df['drug_b_lower'] = self.df['drug_b'].astype(str).str.lower().str.strip()
            print("[INFO] RAG Knowledge Base initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize RAG system: {e}")
            self.df = None
    def search_interaction(self, drug1, drug2):
        if self.df is None: return None
        d1_lower = drug1.lower().strip(); d2_lower = drug2.lower().strip()
        for _, row in self.df.iterrows():
            row_a = row['drug_a_lower']; row_b = row['drug_b_lower']
            if (d1_lower in row_a and d2_lower in row_b) or \
               (d1_lower in row_b and d2_lower in row_a):
                return row.to_dict()
        return None
rag_system = RAGSystem('interactions.csv')

# --- Helper & AI Functions (Unchanged) ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn
def calculate_age(dob_str):
    if not dob_str: return None
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError): return None
def ask_local_llm(context):
    prompt = f"""You are an expert clinical pharmacologist AI. Your persona is that of a knowledgeable, caring, and direct friend. Analyze the complete [CONTEXT] provided and generate a response.

    [CONTEXT]
    {context}

    [INSTRUCTIONS]
    1.  **Analyze Holistically:** Review all information: the patient's profile, all current medications, the new drug, and crucially, the **ADE Risk Score Prediction**.
    2.  **Synthesize Findings:** Generate a single, easy-to-understand summary.
        * **If the ADE Risk Score is HIGH (e.g., >60%)**: Start by highlighting this. **Example:** "Hi Ganesh, I've analyzed your full profile and my prediction model shows a high 75% risk of an adverse event with this new combination. This is mainly because..."
        * **If any CRITICAL pairwise interaction is found:** State it clearly. **Example:** "...this is a major concern. Taking an NSAID while you are on Warfarin is very dangerous. Please do not take this."
    3.  **Final Verdict:** End your response with a clear, one-line verdict on a new line: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD".
    """
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException:
        return "I am unable to connect to the AI assistant. Please ensure LM Studio is running.\nVerdict: DO NOT ADD"

# --- USER AUTH & PROFILE (Unchanged) ---
@app.route('/')
def home(): return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'patient_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form['name']; password = request.form['password']
        conn = get_db_connection(); patient = conn.execute('SELECT * FROM patients WHERE name = ?', (name,)).fetchone(); conn.close()
        if patient and check_password_hash(patient['password_hash'], password):
            session['patient_id'] = patient['id']; return redirect(url_for('dashboard'))
        else: flash("Invalid username or password.", "warning")
    return render_template('index.html', page='login')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']; password = request.form['password']
        conn = get_db_connection()
        if conn.execute('SELECT id FROM patients WHERE name = ?', (name,)).fetchone():
            flash("A patient with this name already exists.", "warning"); conn.close(); return redirect(url_for('register'))
        password_hash = generate_password_hash(password)
        cursor = conn.cursor(); cursor.execute('INSERT INTO patients (name, password_hash) VALUES (?, ?)', (name, password_hash)); conn.commit()
        new_patient = conn.execute('SELECT * FROM patients WHERE name = ?', (name,)).fetchone(); conn.close()
        session['patient_id'] = new_patient['id']
        flash("Registration successful! Please complete your comprehensive health profile.", "info")
        return redirect(url_for('profile'))
    return render_template('index.html', page='register')
@app.route('/dashboard')
def dashboard():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection(); patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    patient = dict(patient_data); patient['age'] = calculate_age(patient.get('dob'))
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient, medications=medications)
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    if request.method == 'POST':
        form_data = (request.form['dob'], request.form['gender'], request.form['weight_kg'], request.form['height_cm'], request.form['emergency_contact'], request.form['conditions'], request.form['drug_allergies'], request.form['food_allergies'], request.form['other_allergies'], request.form['is_smoker'], request.form['alcohol_consumption'], session['patient_id'])
        conn.execute('UPDATE patients SET dob=?, gender=?, weight_kg=?, height_cm=?, emergency_contact=?, conditions=?, drug_allergies=?, food_allergies=?, other_allergies=?, is_smoker=?, alcohol_consumption=? WHERE id=?', form_data)
        conn.commit(); conn.close()
        flash("Profile updated successfully!", "success"); return redirect(url_for('dashboard'))
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone(); conn.close()
    return render_template('index.html', page='profile', patient=patient)

# --- HOLISTIC CHECKING LOGIC WITH PREDICTION ---
@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    if 'patient_id' not in session: return jsonify({'error': 'User not logged in'}), 401
    if not ade_predictor: return jsonify({'summary': "Prediction model not loaded. Cannot perform check.", 'can_add': False})
    
    new_drug = request.json['drug_name']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    existing_meds = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()
    
    # 1. Prepare Features for the ML Model
    age = calculate_age(patient['dob']) or 0
    num_conditions = len(patient['conditions'].split(',')) if patient['conditions'] else 0
    num_allergies = len(patient['drug_allergies'].split(',')) if patient['drug_allergies'] else 0
    num_meds = len(existing_meds) + 1 # Include the new drug in the count
    
    features = np.array([[age, num_conditions, num_allergies, num_meds]])
    
    # 2. Get Prediction from the ML Model
    ade_risk_prob = ade_predictor.predict_proba(features)[0][1] # Probability of class 1 (ADE)
    ade_risk_percent = int(ade_risk_prob * 100)

    # 3. Get Pairwise Interactions
    pairwise_risks = []
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction:
            pairwise_risks.append(f"- {interaction['drug_a']} & {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}")
            
    # 4. Build Holistic Context for the LLM
    context_for_llm = f"Patient: {patient['name']}, Age: {age}.\n"
    context_for_llm += f"Conditions: {patient['conditions']}.\nAllergies: {patient['drug_allergies']}.\n"
    context_for_llm += f"Current Meds: {', '.join([m['drug_name'] for m in existing_meds])}.\n"
    context_for_llm += f"New Drug: {new_drug}.\n"
    context_for_llm += f"ML Model Prediction: {ade_risk_percent}% risk of Adverse Drug Event.\n"
    if pairwise_risks: context_for_llm += "Specific Interactions Found:\n" + "\n".join(pairwise_risks)
        
    # 5. Get Final Analysis from the LLM
    ai_summary = ask_local_llm(context_for_llm)
    
    summary_lines = ai_summary.split('\n')
    verdict = summary_lines[-1]
    can_add = "SAFE TO ADD" in verdict
    main_summary = "\n".join(summary_lines[:-1])

    return jsonify({'summary': main_summary, 'can_add': can_add})

@app.route('/add_medication', methods=['POST'])
def add_medication():
    if 'patient_id' not in session: return redirect(url_for('login'))
    new_drug = request.form['drug_name']; dosage = request.form['dosage']
    conn = get_db_connection()
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)', (session['patient_id'], new_drug, dosage))
    conn.commit(); conn.close()
    flash(f"'{new_drug.title()}' has been added to your log.", "success")
    return redirect(url_for('dashboard'))

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    # This can be simplified now or expanded to also use the ML model
    # For now, keeping it as is for general questions
    if 'patient_id' not in session: return redirect(url_for('login'))
    question = request.form['question']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    # ... rest of the logic for the general assistant
    # ...
    conn.close()
    # This needs to be filled out similar to the check_before_adding if you want prediction here too
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.pop('patient_id', None); flash("You have been logged out.", "info"); return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

