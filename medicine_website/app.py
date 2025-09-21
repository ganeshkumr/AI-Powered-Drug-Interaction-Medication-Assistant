from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'the_final_and_most_secure_key' 

# --- Load the Prediction Model "Brain" ---
try:
    ade_predictor = joblib.load('models/ade_predictor.joblib')
    print("[INFO] Adverse Drug Event (ADE) prediction model loaded successfully.")
except FileNotFoundError:
    print("[ERROR] 'models/ade_predictor.joblib' not found. Please run train_model.py first.")
    ade_predictor = None

# --- RAG System ---
class RAGSystem:
    def __init__(self, data_file):
        try:
            self.df = pd.read_csv(data_file).fillna('')
            self.df['drug_a_lower'] = self.df['drug_a'].astype(str).str.lower().str.strip()
            self.df['drug_b_lower'] = self.df['drug_b'].astype(str).str.lower().str.strip()
            print("[INFO] RAG Knowledge Base initialized successfully.")
        except Exception as e: print(f"[ERROR] Could not initialize RAG system: {e}"); self.df = None
    def search_interaction(self, drug1, drug2):
        if self.df is None: return None
        d1_lower = drug1.lower().strip(); d2_lower = drug2.lower().strip()
        for _, row in self.df.iterrows():
            row_a = row['drug_a_lower']; row_b = row['drug_b_lower']
            if (d1_lower in row_a and d2_lower in row_b) or \
               (d1_lower in row_b and d2_lower in row_a): return row.to_dict()
        return None
rag_system = RAGSystem('interactions.csv')

# --- Helper Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn
def calculate_age(dob_str):
    if not dob_str: return 0
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError): return 0

# --- THIS IS THE UPGRADED AI PROMPT ---
def ask_local_llm(context):
    prompt = f"""You are a personal AI health assistant. Your persona is that of a knowledgeable, caring, and direct friend who prioritizes the user's safety. Your primary goal is to provide a DETAILED and CLEAR explanation for your safety assessment.

    [CONTEXT]
    {context}

    [INSTRUCTIONS]
    1.  **Adopt a Persona:** Speak directly to the patient by their name. Your tone should be warm, personal, and easy to understand.
    2.  **Explain the 'Why' in Detail (MOST IMPORTANT TASK):**
        * You MUST use the "Specific Interactions Found" from the context to explain *why* there might be a risk.
        * For EACH interaction, you must explicitly state the **consequence** of that interaction. Don't just say "there is a risk"; explain what the risk IS (e.g., "this combination can increase your risk of bleeding," or "it might make your blood pressure drop too low"). This is the most helpful information you can provide.
    3.  **Synthesize Findings:**
        * **Example of a good, detailed explanation:** "Hi Ganesh, I've looked at this for you. My main concern is that my knowledge base shows a serious interaction between Warfarin and Aspirin. Taking these together can **significantly increase your risk of bleeding**. Also, your profile mentions a history of stomach ulcers, and Aspirin can make that condition worse. Because of these clear risks, my advice is not to take this combination."
        * **If no interactions are found:** Be reassuring. **Example:** "Hi Ganesh, I've checked my knowledge base, and I don't see any major interactions listed for this medication with your current regimen or health conditions. It looks to be a safe combination."
    4.  **Final Verdict (if applicable):** If the context is about adding a new drug, end your response with a clear, one-line verdict on a new line: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD".
    """
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
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
    if 'patient_id' not in session: return redirect(url_for('dashboard'))
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
        form_data = (request.form.get('dob'), request.form.get('gender'), request.form.get('weight_kg'), request.form.get('height_cm'), request.form.get('emergency_contact'), request.form.get('conditions'), request.form.get('drug_allergies'), request.form.get('food_allergies'), request.form.get('other_allergies'), request.form.get('is_smoker'), request.form.get('alcohol_consumption'), session['patient_id'])
        conn.execute('UPDATE patients SET dob=?, gender=?, weight_kg=?, height_cm=?, emergency_contact=?, conditions=?, drug_allergies=?, food_allergies=?, other_allergies=?, is_smoker=?, alcohol_consumption=? WHERE id=?', form_data)
        conn.commit(); conn.close()
        flash("Profile updated successfully!", "success"); return redirect(url_for('dashboard'))
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone(); conn.close()
    return render_template('index.html', page='profile', patient=patient)

# --- DECOUPLED CHECKING LOGIC ---
@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    if 'patient_id' not in session: return jsonify({'error': 'User not logged in'}), 401
    
    new_drug = request.json['drug_name']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    existing_meds = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()

    # --- XGBoost Prediction ---
    risk_percent = 0
    if ade_predictor:
        age = calculate_age(patient['dob']) or 0
        num_conditions = len(patient['conditions'].split(',')) if patient['conditions'] else 0
        num_allergies = len(patient['drug_allergies'].split(',')) if patient['drug_allergies'] else 0
        num_meds = len(existing_meds) + 1
        features = np.array([[age, num_conditions, num_allergies, num_meds]])
        ade_risk_prob = ade_predictor.predict_proba(features)[0][1]
        risk_percent = int(ade_risk_prob * 100)

    # --- RAG Interaction Search ---
    pairwise_risks = []
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction: pairwise_risks.append(f"- {interaction['drug_a']} & {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}")
    
    # --- LLM Explanation ---
    context_for_llm = f"Patient: {patient['name']}.\n"
    if pairwise_risks: 
        context_for_llm += "Specific Interactions Found:\n" + "\n".join(pairwise_risks)
    else:
        context_for_llm += "Specific Interactions Found: None in the knowledge base.\n"
    
    explanation = ask_local_llm(context_for_llm)

    # The medication is not safe if the risk is high OR there are any interactions
    can_add = risk_percent < 50 and not pairwise_risks

    return jsonify({'risk_percent': risk_percent, 'explanation': explanation, 'can_add': can_add})

@app.route('/add_medication', methods=['POST'])
def add_medication():
    if 'patient_id' not in session: return redirect(url_for('login'))
    form_data = (session['patient_id'], request.form['drug_name'], request.form.get('dosage_amount'), request.form.get('dosage_unit'), request.form.get('frequency'), request.form.get('start_date'), request.form.get('end_date'))
    conn = get_db_connection(); conn.execute('INSERT INTO medications (patient_id, drug_name, dosage_amount, dosage_unit, frequency, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)', form_data); conn.commit(); conn.close()
    flash(f"'{request.form['drug_name']}' has been added to your log.", "success"); return redirect(url_for('dashboard'))

# --- RESTORED AND UPGRADED AI ASSISTANT ---
@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    question = request.form['question']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    
    # Try to extract a specific drug from the user's question for a targeted check
    match = re.search(r'(take|about|check) ([\w\s-]+)\?*$', question.lower())
    topic_to_check = match.group(2).strip() if match else question

    # Build the complete, holistic context
    context_str = f"Patient Profile: Name: {patient['name']}, Age: {calculate_age(patient['dob'])}, Conditions: {patient['conditions']}, Allergies: {patient['drug_allergies']}.\n"
    context_str += f"Current Medications: {', '.join([med['drug_name'] for med in existing_meds]) if existing_meds else 'None'}.\n"
    context_str += f"Patient's Question is about: {topic_to_check}.\n\n"
    context_str += "Specific Interactions Found in Knowledge Base:\n"
    
    # Check topic against profile and all existing meds
    all_risks = get_personalized_warnings(topic_to_check, patient)
    for med in existing_meds:
        interaction = rag_system.search_interaction(topic_to_check, med['drug_name'])
        if interaction:
            all_risks.append(f"- {interaction['drug_a']} & {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}")
            
    if all_risks:
        context_str += "\n".join(all_risks)
    else:
        context_str += "None relevant to the question."

    # Get the AI's final, detailed explanation
    ai_response = ask_local_llm(context_str)
    
    # Re-render the dashboard with the new response
    patient_dict = dict(patient); patient_dict['age'] = calculate_age(patient.get('dob'))
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient_dict, medications=medications, ai_response=ai_response)

@app.route('/logout')
def logout():
    session.pop('patient_id', None); flash("You have been logged out.", "info"); return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

