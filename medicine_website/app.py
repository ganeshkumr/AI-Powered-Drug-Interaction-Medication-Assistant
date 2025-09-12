from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date

app = Flask(__name__)
app.secret_key = 'the_final_and_most_secure_key' 

# --- RAG System: The Single Source of Truth ---
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

# --- Helper Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn

def calculate_age(dob_str):
    if not dob_str: return None
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError):
        return None

# --- AI & Safety Check Logic ---
def ask_local_llm(context):
    prompt = f"""You are an expert clinical pharmacologist AI. Your persona is that of a knowledgeable, caring, and direct friend. Analyze the complete [CONTEXT] provided and generate a response based on the [INSTRUCTIONS].

    [CONTEXT]
    {context}

    [INSTRUCTIONS]
    1.  **Adopt a Persona:** Speak directly to the patient by name. Be warm and personal, but become very firm when a serious risk is detected.
    2.  **Analyze Holistically:** This is your most important task. Review all the information provided: the patient's profile, their **entire list of current medications**, and the new drug they are asking about. Look for not just pairwise (A+B) interactions, but also potential multi-drug (A+B+C) interactions based on your internal knowledge.
    3.  **Synthesize Findings:** Generate a single, easy-to-understand summary.
        * **If any CRITICAL or HIGH RISK is found:** State it clearly and directly. Explain the risk in simple terms. **Example:** "Hi Ganesh, I've looked at your full medication list and this is a major concern. Taking an NSAID while you are on both Warfarin and Methotrexate significantly increases the risk of severe side effects. Please do not take this. It's really important to talk to your doctor about a safer alternative."
        * **If MODERATE risks are found:** Explain them simply. **Example:** "Okay Ganesh, let's look at this. Taking an ACE inhibitor and you want to add a potassium supplement. Taking these together can sometimes raise your potassium levels too high. I'd really recommend talking to your doctor about this first."
        * **If everything is SAFE:** Be reassuring. **Example:** "Hi Ganesh, I've checked Metformin against your profile and all your other medications, and everything looks safe. It's great that you're being so careful!"
    4.  **Final Verdict:** End your response with a clear, one-line verdict on a new line: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD". This is mandatory.
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

# --- USER AUTHENTICATION & PROFILE ---
@app.route('/')
def home(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'patient_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form['name']; password = request.form['password']
        conn = get_db_connection()
        patient = conn.execute('SELECT * FROM patients WHERE name = ?', (name,)).fetchone()
        conn.close()
        if patient and check_password_hash(patient['password_hash'], password):
            session['patient_id'] = patient['id']
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "warning")
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

# --- MAIN DASHBOARD & LOGIC ---
@app.route('/dashboard')
def dashboard():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    # Convert Row object to a mutable dictionary
    patient = dict(patient_data)
    patient['age'] = calculate_age(patient.get('dob')) # .get() works here because it's a dict
    
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient, medications=medications)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    if request.method == 'POST':
        # Collect all form data
        form_data = (
            request.form['dob'], request.form['gender'], request.form['weight_kg'], request.form['height_cm'],
            request.form['emergency_contact'], request.form['conditions'], request.form['drug_allergies'],
            request.form['food_allergies'], request.form['other_allergies'], request.form['is_smoker'],
            request.form['alcohol_consumption'], session['patient_id']
        )
        conn.execute('''
            UPDATE patients SET dob=?, gender=?, weight_kg=?, height_cm=?, emergency_contact=?, 
            conditions=?, drug_allergies=?, food_allergies=?, other_allergies=?, is_smoker=?, 
            alcohol_consumption=? WHERE id=?
        ''', form_data)
        conn.commit(); conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('dashboard'))
        
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    return render_template('index.html', page='profile', patient=patient)

# --- HOLISTIC CHECKING LOGIC ---
def get_holistic_context(new_drug, patient, existing_meds):
    """Builds a complete context string for the AI to analyze."""
    # --- THIS IS THE FIX ---
    # Access sqlite3.Row objects using dictionary-style keys, not the .get() method.
    patient_age = calculate_age(patient['dob'])
    context_str = f"Patient Profile: Name: {patient['name']}, Age: {patient_age}, Gender: {patient['gender']}, Weight: {patient['weight_kg']}kg, Height: {patient['height_cm']}cm.\n"
    context_str += f"Lifestyle: Smoker: {patient['is_smoker']}, Alcohol: {patient['alcohol_consumption']}.\n"
    context_str += f"Conditions: {patient['conditions']}.\n"
    context_str += f"Allergies: Drug: {patient['drug_allergies']}, Food: {patient['food_allergies']}, Other: {patient['other_allergies']}.\n"
    context_str += f"Current Medications: {', '.join([med['drug_name'] for med in existing_meds]) if existing_meds else 'None'}.\n"
    context_str += f"New Drug to Analyze: {new_drug}.\n\n"
    context_str += "Known Pairwise Interactions from Database:\n"
    
    found_pairwise = False
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction:
            found_pairwise = True
            context_str += f"- {interaction['drug_a']} and {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}\n"
            
    if not found_pairwise:
        context_str += "- No specific pairwise interactions were found in the knowledge base for the new drug against the current log.\n"
        
    return context_str

@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    if 'patient_id' not in session: return jsonify({'error': 'User not logged in'}), 401
    
    new_drug = request.json['drug_name']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    existing_meds = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()
    
    holistic_context = get_holistic_context(new_drug, patient, existing_meds)
    ai_summary = ask_local_llm(holistic_context)

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
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    question = request.form['question']
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    
    # Pass the raw sqlite3.Row object directly to the context function
    holistic_context = get_holistic_context(question, patient, existing_meds)
    ai_response = ask_local_llm(holistic_context)

    # Re-fetch and convert to dict for the template
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    patient_dict = dict(patient_data)
    patient_dict['age'] = calculate_age(patient_dict.get('dob'))
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient_dict, medications=medications, ai_response=ai_response.split('\nVerdict:')[0])

@app.route('/logout')
def logout():
    session.pop('patient_id', None); flash("You have been logged out.", "info"); return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

