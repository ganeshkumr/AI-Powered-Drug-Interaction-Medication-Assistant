from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

app = Flask(__name__)
app.secret_key = 'the_final_and_most_secure_key' 

# --- GNN Model Definition and Loading ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels):
        super(GNNLinkPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden_channels, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=0.6)

    def encode(self, x, edge_index):
        x = self.embedding(x); x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index)); x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index); return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]; dst = z[edge_label_index[1]]; return (src * dst).sum(dim=-1)

def load_gnn_model():
    try:
        with open('models/drug_map.json', 'r') as f: drug_map = json.load(f)
        model = GNNLinkPredictor(num_nodes=len(drug_map), embedding_dim=128, hidden_channels=128, out_channels=128)
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load('models/gnn_model.pt', map_location=map_location))
        model.eval(); print("[INFO] GNN Prediction Model loaded successfully on CPU.")
        return model, drug_map
    except FileNotFoundError:
        print("[ERROR] GNN model not found. Please run train_gnn.py first.")
        return None, None
gnn_model, drug_map = load_gnn_model()

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

# --- Helper, Validation & AI Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn
def calculate_age(dob_str):
    if not dob_str: return 0
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError): return 0
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)
def is_strong_password(password):
    if len(password) < 8: return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password): return False, "Password must contain an uppercase letter."
    if not re.search(r"[a-z]", password): return False, "Password must contain a lowercase letter."
    if not re.search(r"[0-9]", password): return False, "Password must contain a number."
    if not re.search(r"[!@#$%^&*(),.?:{}|<>]", password): return False, "Password must contain a special character."
    return True, ""
def ask_local_llm(context):
    prompt = f"""You are a personal AI health assistant. Your persona is that of a knowledgeable, caring, and direct friend. Your primary goal is to provide a DETAILED and CLEAR explanation for your safety assessment.

    [CONTEXT]
    {context}

    [INSTRUCTIONS]
    1.  *Adopt a Persona:* Speak directly to the patient by their name. Your tone should be warm, personal, and easy to understand. Do NOT use technical jargon like "database" or "entities".
    2.  *Integrate the GNN Prediction:* If a "GNN Predicted Risk" is in the context, state it naturally in your explanation.
    3.  *Explain the 'Why' in Detail (MOST IMPORTANT TASK):*
        * You MUST use the "Factual Interactions from Knowledge Base" from the context to explain why there might be a risk.
        * For EACH interaction, you must explicitly state the *consequence* of that interaction. Don't just say "there is a risk"; explain what the risk IS (e.g., "this combination can increase your risk of bleeding," or "it might make your blood pressure drop too low"). This is the most helpful information you can provide.
    4.  *Synthesize Findings:*
        * *Example of a good, detailed explanation:* "Hi Ganesh, I looked this up for you. My analysis predicts a high interaction risk of about 92% here. My main concern is that my information shows a serious interaction between Warfarin and Ibuprofen, which can *significantly increase your risk of bleeding*. Because of this clear risk, my advice is not to take this combination."
        * *If no interactions are found:* Be reassuring. *Example:* "Hi Ganesh, I've checked my information, and I don't see any major interactions listed for this medication with your current regimen. The predicted risk is also very low, so this looks safe."
    5.  *Final Verdict:* End your response with a clear, one-line verdict on a new line: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD".
    """
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException:
        return "I am unable to connect to the AI assistant.\nVerdict: DO NOT ADD"

# --- USER AUTHENTICATION & PROFILE (UPGRADED) ---
@app.route('/')
def home(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'patient_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
        conn.close()
        if patient and check_password_hash(patient['password_hash'], password):
            session['patient_id'] = patient['id']
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password. Please try again.", "warning")
    return render_template('index.html', page='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        if not is_valid_email(email):
            flash("Please enter a valid email address.", "warning")
            return redirect(url_for('register'))
            
        is_strong, message = is_strong_password(password)
        if not is_strong:
            flash(message, "warning")
            return redirect(url_for('register'))
        
        conn = get_db_connection()
        if conn.execute('SELECT id FROM patients WHERE email = ?', (email,)).fetchone():
            flash("An account with this email already exists. Please login.", "warning")
            conn.close()
            return redirect(url_for('login'))
        
        password_hash = generate_password_hash(password)
        # Correctly insert all required fields
        conn.execute('INSERT INTO patients (name, email, password_hash) VALUES (?, ?, ?)', (name, email, password_hash))
        conn.commit()
        new_patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
        conn.close()
        
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
    patient = dict(patient_data)
    patient['age'] = calculate_age(patient.get('dob'))
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient, medications=medications)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    if request.method == 'POST':
        form_data = (
            request.form.get('dob', ''), request.form.get('gender', ''), request.form.get('weight_kg', None), 
            request.form.get('height_cm', None), request.form.get('emergency_contact', ''), 
            request.form.get('conditions', ''), request.form.get('drug_allergies', ''), 
            request.form.get('food_allergies', ''), request.form.get('other_allergies', ''), 
            request.form.get('is_smoker', ''), request.form.get('alcohol_consumption', ''), 
            session['patient_id']
        )
        conn.execute('UPDATE patients SET dob=?, gender=?, weight_kg=?, height_cm=?, emergency_contact=?, conditions=?, drug_allergies=?, food_allergies=?, other_allergies=?, is_smoker=?, alcohol_consumption=? WHERE id=?', form_data)
        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('dashboard'))
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    return render_template('index.html', page='profile', patient=patient)

# ... (The rest of your app.py file, including check_before_adding, add_medication, etc., remains the same)
def get_holistic_context(new_drug, patient, existing_meds):
    patient_age = calculate_age(patient.get('dob'))
    context_str = f"Patient Profile: Name: {patient.get('name')}, Age: {patient_age}, Conditions: {patient.get('conditions')}, Allergies: {patient.get('drug_allergies')}, Smoker: {patient.get('is_smoker')}, Alcohol: {patient.get('alcohol_consumption')}.\n"
    context_str += f"Current Medications: {', '.join([med['drug_name'] for med in existing_meds]) if existing_meds else 'None'}.\n"
    context_str += f"New Drug to Analyze: {new_drug}.\n\n"
    context_str += "Known Pairwise Interactions from Database (for reference):\n"
    found_pairwise = False
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction:
            found_pairwise = True
            context_str += f"- {interaction['drug_a']} and {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}\n"
    if not found_pairwise: 
        context_str += "- No specific pairwise interactions were found in the knowledge base.\n"
    return context_str

@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    if 'patient_id' not in session: return jsonify({'error': 'User not logged in'}), 401
    
    new_drug = request.json['drug_name']
    conn = get_db_connection()
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    patient = dict(patient_data)
    existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
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
    form_data = (session['patient_id'], request.form['drug_name'], request.form.get('dosage_amount'), request.form.get('dosage_unit'), request.form.get('frequency'), request.form.get('start_date'), request.form.get('end_date'))
    conn = get_db_connection()
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage_amount, dosage_unit, frequency, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)', form_data)
    conn.commit()
    conn.close()
    flash(f"'{request.form['drug_name']}' has been added to your log.", "success")
    return redirect(url_for('dashboard'))

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    question = request.form['question']
    conn = get_db_connection()
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    patient = dict(patient_data)
    existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    
    new_drug_match = re.search(r'(take|about|check) ([\w\s-]+)\?*$', question.lower())
    topic_to_check = new_drug_match.group(2).strip() if new_drug_match else question

    holistic_context = get_holistic_context(topic_to_check, patient, existing_meds)
    ai_response = ask_local_llm(holistic_context)
    
    patient['age'] = calculate_age(patient.get('dob'))
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient, medications=medications, ai_response=ai_response.split('\nVerdict:')[0])

@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

