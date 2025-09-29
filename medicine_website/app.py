from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date
import torch
from torch_geometric.nn import SAGEConv

app = Flask(__name__)
app.secret_key = 'the_final_and_most_secure_key' 

# --- GNN Model Definition and Loading ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super(GNNLinkPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, embedding_dim * 2)
        self.conv2 = SAGEConv(embedding_dim * 2, embedding_dim)
    def encode(self, x, edge_index):
        x = self.embedding(x); x = self.conv1(x, edge_index).relu(); x = self.conv2(x, edge_index); return x
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]; dst = z[edge_label_index[1]]; return (src * dst).sum(dim=-1)

def load_gnn_model():
    try:
        with open('models/drug_map.json', 'r') as f: drug_map = json.load(f)
        model = GNNLinkPredictor(num_nodes=len(drug_map))
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load('models/gnn_model.pt', map_location=map_location))
        model.eval()
        print("[INFO] GNN Prediction Model loaded successfully on CPU.")
        return model, drug_map
    except FileNotFoundError:
        print("[ERROR] GNN model not found. Please run train_gnn.py first.")
        return None, None
gnn_model, drug_map = load_gnn_model()

# --- RAG System (Used for factual lookup) ---
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

# --- Helper & AI Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn
def calculate_age(dob_str):
    if not dob_str: return 0
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError): return 0

def predict_gnn_risk(drug1, drug2):
    if not gnn_model or not drug_map: return 0.0
    d1_idx = drug_map.get(drug1.strip()); d2_idx = drug_map.get(drug2.strip())
    if d1_idx is None or d2_idx is None: return 0.0
    with torch.no_grad():
        embeddings = gnn_model.embedding.weight
        d1_emb = embeddings[d1_idx]; d2_emb = embeddings[d2_idx]
        score = torch.sigmoid((d1_emb * d2_emb).sum()).item()
        return score

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
        * **Example of a good, detailed explanation:** "Hi Ganesh, I looked this up for you. My knowledge base shows a serious interaction between Warfarin and Aspirin, which can **significantly increase your risk of bleeding**. Also, your profile mentions a history of stomach ulcers, and Aspirin can make that condition worse. Because of these clear risks, my advice is not to take this combination."
        * **If no interactions are found:** Be reassuring. **Example:** "Hi Ganesh, I've checked my knowledge base, and I don't see any major interactions listed for this medication with your current regimen or health conditions. It looks to be a safe combination."
    """
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException:
        return "I am unable to connect to the AI assistant to provide an explanation. Please ensure LM Studio is running."

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
    existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()

    # --- 1. GNN Prediction (Quantitative) ---
    risk_scores = []
    for med in existing_meds:
        # We need to pass the string from the DB row
        risk = predict_gnn_risk(new_drug, med['drug_name'])
        risk_scores.append(risk)
    overall_risk_percent = int(max(risk_scores) * 100) if risk_scores else 0

    # --- 2. RAG + LLM Explanation (Qualitative) ---
    rag_facts = []
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction:
            rag_facts.append(f"- {interaction['drug_a']} & {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}")
    
    context_for_llm = f"Patient: {patient['name']}.\n"
    if rag_facts: 
        context_for_llm += "Specific Interactions Found:\n" + "\n".join(rag_facts)
    else:
        context_for_llm += "Specific Interactions Found: None in the knowledge base.\n"
    
    explanation = ask_local_llm(context_for_llm)

    # A drug is unsafe if the GNN predicts a high risk OR if the RAG finds a known interaction.
    can_add = overall_risk_percent < 70 and not rag_facts

    return jsonify({'risk_percent': overall_risk_percent, 'explanation': explanation, 'can_add': can_add})

@app.route('/add_medication', methods=['POST'])
def add_medication():
    if 'patient_id' not in session: return redirect(url_for('login'))
    form_data = (session['patient_id'], request.form['drug_name'], request.form.get('dosage_amount'), request.form.get('dosage_unit'), request.form.get('frequency'), request.form.get('start_date'), request.form.get('end_date'))
    conn = get_db_connection(); conn.execute('INSERT INTO medications (patient_id, drug_name, dosage_amount, dosage_unit, frequency, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)', form_data); conn.commit(); conn.close()
    flash(f"'{request.form['drug_name']}' has been added to your log.", "success"); return redirect(url_for('dashboard'))

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    return redirect(url_for('dashboard')) # Simplified for now

@app.route('/logout')
def logout():
    session.pop('patient_id', None); flash("You have been logged out.", "info"); return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

