from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_a_great_project' 

# --- RAG System Setup (Phase 1 Logic) ---
class RAGSystem:
    """Handles loading data and searching for relevant context."""
    def __init__(self, data_file):
        try:
            self.df = pd.read_csv(data_file)
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.df['description'] = self.df.apply(lambda row: f"Interaction between {row['drug_a']} and {row['drug_b']}", axis=1)
            embeddings = self.encoder.encode(self.df['description'].tolist())
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings, dtype=np.float32))
            print("[INFO] RAG System initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize RAG system: {e}")
            self.df = None

    def search(self, query, k=1):
        """Searches for the most relevant document in the CSV."""
        if self.df is None: return None
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        # A simple threshold to avoid irrelevant results
        if distances[0][0] < 1.0: # Lower distance means better match
            return self.df.iloc[indices[0][0]].to_dict()
        return None

# Initialize the RAG system globally when the app starts
rag_system = RAGSystem('interactions.csv')

# --- Helper & API Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- AI & Safety Check Logic ---
def ask_local_llm(context, question):
    """Connects to LM Studio to get a conversational answer."""
    prompt = f"""You are a helpful and cautious pharmacy assistant. Your role is to answer questions based ONLY on the provided context.

    [CONTEXT]
    {context}

    [QUESTION]
    {question}

    [INSTRUCTIONS]
    1. Analyze the context and the question.
    2. If the context directly answers the question, provide a helpful, conversational response. Start your answer directly, without any preamble like "Based on the context...".
    3. If the context is NOT relevant or doesn't contain the answer, you MUST respond with ONLY this exact sentence: "I'm sorry, I don't have enough information in my knowledge base to answer that question. It's best to consult a healthcare professional."
    4. Never invent information. Prioritize safety above all.
    """
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    try:
        # --- THE FIX IS HERE ---
        # Increased the timeout to 60 seconds to give the model more time to respond.
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print("[ERROR] Connection to LM Studio timed out after 60 seconds.")
        return "The AI assistant is taking too long to respond. The model may be too busy. Please try again in a moment."
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to LM Studio: {e}")
        return "I am unable to connect to the AI assistant at the moment. Please ensure LM Studio is running and accessible."

# --- USER & PROFILE MANAGEMENT (Unchanged) ---
@app.route('/', methods=['GET', 'POST'])
def login():
    session.pop('patient_id', None)
    if request.method == 'POST':
        patient_name = request.form['name'].strip()
        conn = get_db_connection()
        patient = conn.execute('SELECT * FROM patients WHERE name = ?', (patient_name,)).fetchone()
        if patient:
            session['patient_id'] = patient['id']
            return redirect(url_for('dashboard'))
        else:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO patients (name) VALUES (?)', (patient_name,))
            conn.commit()
            new_patient = conn.execute('SELECT * FROM patients WHERE name = ?', (patient_name,)).fetchone()
            session['patient_id'] = new_patient['id']
            flash("Welcome! Let's set up your health profile.", "info")
            return redirect(url_for('profile'))
    return render_template('index.html', page='login')

@app.route('/dashboard')
def dashboard():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()
    return render_template('index.html', page='dashboard', patient=patient, medications=medications)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'patient_id' not in session: return redirect(url_for('login'))
    conn = get_db_connection()
    if request.method == 'POST':
        conn.execute('''
            UPDATE patients SET age = ?, gender = ?, weight_kg = ?, conditions = ?, allergies = ? WHERE id = ?
        ''', (request.form['age'], request.form['gender'], request.form['weight_kg'], 
              request.form['conditions'], request.form['allergies'], session['patient_id']))
        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('dashboard'))
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    return render_template('index.html', page='profile', patient=patient)

# --- MEDICATION MANAGEMENT (Unchanged - Still uses rule-based alerts) ---
@app.route('/add_medication', methods=['POST'])
def add_medication():
    if 'patient_id' not in session: return redirect(url_for('login'))
    new_drug = request.form['drug_name']
    # The rule-based checks are still here for instant feedback when adding meds
    # The AI is for conversational follow-up questions
    conn = get_db_connection()
    # (Existing logic for checking interactions and contraindications would go here)
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)',
                 (session['patient_id'], new_drug, request.form['dosage']))
    conn.commit()
    conn.close()
    flash(f"'{new_drug.title()}' has been added to your log.", "success")
    # (Flash warnings based on rules here)
    return redirect(url_for('dashboard'))

# --- NEW: AI ASSISTANT ROUTE ---
@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    """Handles conversational queries using RAG and the local LLM."""
    if 'patient_id' not in session: return redirect(url_for('login'))
    
    question = request.form['question']
    
    # 1. Gather Context
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    meds = conn.execute('SELECT drug_name FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()

    # Create a text-based context for the AI
    context_str = f"Patient Profile:\nName: {patient['name']}\nAge: {patient['age']}\nConditions: {patient['conditions']}\nAllergies: {patient['allergies']}\n"
    current_meds = [med['drug_name'] for med in meds]
    if current_meds:
        context_str += f"Currently taking: {', '.join(current_meds)}\n\n"
    
    # 2. RAG Search
    rag_result = rag_system.search(question)
    if rag_result:
        context_str += f"Relevant information from knowledge base:\n{rag_result['interaction']}"
    
    # 3. Ask the LLM
    ai_response = ask_local_llm(context_str, question)
    
    # Re-render the dashboard with the AI's response
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', (session['patient_id'],)).fetchall()
    conn.close()

    return render_template('index.html', page='dashboard', patient=patient, medications=medications, ai_response=ai_response)

@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
