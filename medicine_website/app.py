from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date, datetime, timedelta
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Fit OAuth2 imports (optional package)
try:
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    GOOGLE_FIT_AVAILABLE = True
except ImportError:
    GOOGLE_FIT_AVAILABLE = False
    print("[WARNING] google-auth-oauthlib not installed. Google Fit features disabled. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

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

# --- Drug Dosage Validation System ---
class DosageValidator:
    def __init__(self, dosage_file):
        try:
            with open(dosage_file, 'r') as f:
                self.dosage_limits = json.load(f)
            print("[INFO] Drug dosage limits database initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize dosage validator: {e}")
            self.dosage_limits = {}
    
    def validate_dosage(self, drug_name, dosage_amount, dosage_unit, frequency):
        """Validate drug dosage against maximum safe limits"""
        if not drug_name or not dosage_amount:
            return {"is_safe": True, "warnings": [], "max_daily": None, "max_single": None}
        
        drug_key = drug_name.lower().strip()
        
        # Find matching drug in database
        drug_info = None
        for drug, info in self.dosage_limits.items():
            if drug in drug_key or drug_key in drug:
                drug_info = info
                break
        
        if not drug_info:
            return {"is_safe": True, "warnings": ["No dosage limits found for this medication"], "max_daily": None, "max_single": None}
        
        try:
            dosage_amount = float(dosage_amount)
        except (ValueError, TypeError):
            return {"is_safe": False, "warnings": ["Invalid dosage amount"], "max_daily": None, "max_single": None}
        
        warnings = []
        is_safe = True
        
        # Check unit compatibility
        if dosage_unit and dosage_unit.lower() != drug_info["unit"].lower():
            warnings.append(f"Unit mismatch: Expected {drug_info['unit']}, got {dosage_unit}")
        
        # Calculate daily dosage based on frequency
        daily_dosage = self._calculate_daily_dosage(dosage_amount, frequency)
        
        # Check against maximum limits
        max_daily = drug_info["max_daily_mg"] if "max_daily_mg" in drug_info else drug_info.get("max_daily_iu", drug_info.get("max_daily_units", 0))
        max_single = drug_info["max_single_mg"] if "max_single_mg" in drug_info else drug_info.get("max_single_iu", drug_info.get("max_single_units", 0))
        
        if daily_dosage > max_daily:
            warnings.append(f"Daily dosage ({daily_dosage} {drug_info['unit']}) exceeds maximum safe limit ({max_daily} {drug_info['unit']})")
            is_safe = False
        
        if dosage_amount > max_single:
            warnings.append(f"Single dose ({dosage_amount} {drug_info['unit']}) exceeds maximum safe limit ({max_single} {drug_info['unit']})")
            is_safe = False
        
        # Add general warnings from database
        if drug_info.get("warnings"):
            warnings.extend(drug_info["warnings"])
        
        return {
            "is_safe": is_safe,
            "warnings": warnings,
            "max_daily": max_daily,
            "max_single": max_single,
            "daily_dosage": daily_dosage,
            "unit": drug_info["unit"]
        }
    
    def _calculate_daily_dosage(self, single_dose, frequency):
        """Calculate daily dosage based on frequency"""
        if not frequency:
            return single_dose
        
        frequency_lower = frequency.lower()
        
        # Common frequency patterns
        if any(word in frequency_lower for word in ["once", "daily", "day"]):
            return single_dose
        elif any(word in frequency_lower for word in ["twice", "2x", "two"]):
            return single_dose * 2
        elif any(word in frequency_lower for word in ["three", "3x", "thrice"]):
            return single_dose * 3
        elif any(word in frequency_lower for word in ["four", "4x"]):
            return single_dose * 4
        elif any(word in frequency_lower for word in ["every 6", "6 hours"]):
            return single_dose * 4  # 4 times per day
        elif any(word in frequency_lower for word in ["every 8", "8 hours"]):
            return single_dose * 3  # 3 times per day
        elif any(word in frequency_lower for word in ["every 12", "12 hours"]):
            return single_dose * 2  # 2 times per day
        else:
            # Default to single dose if frequency is unclear
            return single_dose

dosage_validator = DosageValidator('drug_dosage_limits.json')

# --- Side Effects Database ---
class SideEffectsDatabase:
    def __init__(self, side_effects_file):
        try:
            with open(side_effects_file, 'r') as f:
                self.side_effects = json.load(f)
            print("[INFO] Side effects database initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize side effects database: {e}")
            self.side_effects = {}
    
    def get_side_effects(self, drug_name, patient_profile=None):
        """Get side effects for a drug with personalized risk assessment"""
        drug_key = drug_name.lower().strip()
        
        # Find matching drug
        drug_info = None
        for drug, info in self.side_effects.items():
            if drug in drug_key or drug_key in drug:
                drug_info = info
                break
        
        if not drug_info:
            return {"side_effects": [], "risk_warnings": []}
        
        side_effects = drug_info.get('common_side_effects', [])
        serious_effects = drug_info.get('serious_side_effects', [])
        risk_factors = drug_info.get('risk_factors', {})
        
        # Generate personalized risk warnings
        risk_warnings = []
        if patient_profile:
            for risk_factor, warning in risk_factors.items():
                if self._check_risk_factor(risk_factor, patient_profile):
                    risk_warnings.append(f"⚠️ {warning}")
        
        return {
            "side_effects": side_effects,
            "serious_effects": serious_effects,
            "risk_warnings": risk_warnings
        }
    
    def _check_risk_factor(self, risk_factor, patient_profile):
        """Check if patient has specific risk factors"""
        if risk_factor == "age_65_plus":
            return patient_profile.get('age', 0) >= 65
        elif risk_factor == "age_60_plus":
            return patient_profile.get('age', 0) >= 60
        elif risk_factor == "age_80_plus":
            return patient_profile.get('age', 0) >= 80
        elif risk_factor == "kidney_disease":
            conditions = patient_profile.get('conditions', '').lower()
            return 'kidney' in conditions or 'renal' in conditions
        elif risk_factor == "liver_disease":
            conditions = patient_profile.get('conditions', '').lower()
            return 'liver' in conditions or 'hepatic' in conditions
        elif risk_factor == "heart_disease":
            conditions = patient_profile.get('conditions', '').lower()
            return 'heart' in conditions or 'cardiac' in conditions
        elif risk_factor == "diabetes":
            conditions = patient_profile.get('conditions', '').lower()
            return 'diabetes' in conditions
        elif risk_factor == "alcohol_use":
            return patient_profile.get('alcohol_consumption', '').lower() in ['regular', 'occasional']
        return False

side_effects_db = SideEffectsDatabase('side_effects_database.json')

# --- Multi-Drug Conflict Checker ---
class MultiDrugConflictChecker:
    def __init__(self, conflicts_file):
        try:
            with open(conflicts_file, 'r') as f:
                self.conflicts_data = json.load(f)
            print("[INFO] Multi-drug conflict checker initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize multi-drug conflict checker: {e}")
            self.conflicts_data = {"triangular_conflicts": [], "category_conflicts": {}}
    
    def check_triangular_conflicts(self, new_drug, existing_meds):
        """Check for 3-drug conflicts (triangular conflicts)"""
        conflicts = []
        
        if len(existing_meds) < 2:
            return conflicts
        
        # Get all drug names
        all_drugs = [med['drug_name'].lower().strip() for med in existing_meds] + [new_drug.lower().strip()]
        
        # Check each triangular conflict
        for conflict in self.conflicts_data.get('triangular_conflicts', []):
            conflict_drugs = [drug.lower().strip() for drug in conflict['drugs']]
            
            # Check if all 3 drugs in conflict are present
            if all(drug in all_drugs for drug in conflict_drugs):
                conflicts.append({
                    'type': 'triangular',
                    'drugs': conflict['drugs'],
                    'severity': conflict['severity'],
                    'description': conflict['description'],
                    'warning': conflict['warning']
                })
        
        return conflicts
    
    def check_category_conflicts(self, new_drug, existing_meds):
        """Check for category-based conflicts"""
        conflicts = []
        
        # Get drug categories
        categories = self.conflicts_data.get('drug_categories', {})
        
        # Find categories for new drug
        new_drug_categories = []
        for category, drugs in categories.items():
            if any(drug in new_drug.lower() for drug in drugs):
                new_drug_categories.append(category)
        
        # Check against existing medications
        for med in existing_meds:
            med_categories = []
            for category, drugs in categories.items():
                if any(drug in med['drug_name'].lower() for drug in drugs):
                    med_categories.append(category)
            
            # Check for category conflicts
            for new_cat in new_drug_categories:
                for existing_cat in med_categories:
                    conflict_key = f"{new_cat}_{existing_cat}"
                    reverse_key = f"{existing_cat}_{new_cat}"
                    
                    if conflict_key in self.conflicts_data.get('category_conflicts', {}):
                        conflict_info = self.conflicts_data['category_conflicts'][conflict_key]
                    elif reverse_key in self.conflicts_data.get('category_conflicts', {}):
                        conflict_info = self.conflicts_data['category_conflicts'][reverse_key]
                    else:
                        continue
                    
                    conflicts.append({
                        'type': 'category',
                        'drugs': [new_drug, med['drug_name']],
                        'severity': conflict_info['severity'],
                        'description': conflict_info['description'],
                        'warning': conflict_info['warning']
                    })
        
        return conflicts

multi_drug_checker = MultiDrugConflictChecker('multi_drug_conflicts.json')

# --- Personalized Dosage Adjustment ---
def calculate_personalized_dosage(drug_name, dosage_amount, patient_profile):
    """Calculate personalized dosage based on patient profile"""
    try:
        dosage_amount = float(dosage_amount)
    except (ValueError, TypeError):
        return {"suggested_dose": dosage_amount, "adjustment_reason": "Invalid dosage amount"}
    
    # Get base dosage limits
    dosage_validation = dosage_validator.validate_dosage(drug_name, dosage_amount, "mg", "daily")
    max_daily = dosage_validation.get('max_daily', dosage_amount)
    
    # Calculate adjustment factors
    adjustment_factor = 1.0
    adjustment_reasons = []
    
    # Age-based adjustments
    age = patient_profile.get('age', 0)
    if age >= 80:
        adjustment_factor *= 0.7
        adjustment_reasons.append("Age 80+ (reduced metabolism)")
    elif age >= 65:
        adjustment_factor *= 0.8
        adjustment_reasons.append("Age 65+ (reduced kidney function)")
    elif age >= 60:
        adjustment_factor *= 0.9
        adjustment_reasons.append("Age 60+ (mild kidney function decline)")
    
    # Condition-based adjustments
    conditions = patient_profile.get('conditions', '').lower()
    
    if 'kidney' in conditions or 'renal' in conditions:
        adjustment_factor *= 0.75
        adjustment_reasons.append("Kidney disease (reduced clearance)")
    
    if 'liver' in conditions or 'hepatic' in conditions:
        adjustment_factor *= 0.8
        adjustment_reasons.append("Liver disease (reduced metabolism)")
    
    if 'heart' in conditions or 'cardiac' in conditions:
        adjustment_factor *= 0.85
        adjustment_reasons.append("Heart disease (reduced tolerance)")
    
    # Weight-based adjustments
    weight = patient_profile.get('weight_kg')
    if weight:
        try:
            weight = float(weight)
            if weight < 50:  # Underweight
                adjustment_factor *= 0.8
                adjustment_reasons.append("Low body weight (reduced tolerance)")
            elif weight > 100:  # Overweight
                adjustment_factor *= 1.1
                adjustment_reasons.append("Higher body weight (increased clearance)")
        except (ValueError, TypeError):
            pass
    
    # Calculate suggested dose
    suggested_dose = min(dosage_amount * adjustment_factor, max_daily)
    
    # Ensure minimum effective dose
    min_effective_dose = max_daily * 0.5  # At least 50% of max dose
    if suggested_dose < min_effective_dose:
        suggested_dose = min_effective_dose
        adjustment_reasons.append("Minimum effective dose maintained")
    
    return {
        "suggested_dose": round(suggested_dose, 1),
        "original_dose": dosage_amount,
        "adjustment_factor": round(adjustment_factor, 2),
        "adjustment_reasons": adjustment_reasons,
        "max_safe_dose": max_daily
    }

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
    2.  *Always Include GNN Prediction:* You MUST include a GNN prediction percentage in your response. If not provided in context, estimate based on the information available (e.g., "My AI analysis shows a 15% risk of interactions" or "The predicted interaction risk is 85%").
    3.  *Explain the 'Why' in Detail (MOST IMPORTANT TASK):*
        * You MUST use the "Factual Interactions from Knowledge Base" from the context to explain why there might be a risk.
        * For EACH interaction, you must explicitly state the *consequence* of that interaction. Don't just say "there is a risk"; explain what the risk IS (e.g., "this combination can increase your risk of bleeding," or "it might make your blood pressure drop too low"). This is the most helpful information you can provide.
    4.  *Dosage Safety Analysis:* If dosage information is provided, explain it in a friendly way:
        * If dosage is safe: "Great news! The dosage you've entered looks safe and within recommended limits."
        * If dosage is too high: "I'm concerned about the dosage - it's higher than what's typically recommended for safety."
        * Always explain the maximum safe doses in simple terms.
    5.  *Synthesize Findings:*
        * *Example of a good, detailed explanation:* "Hi Ganesh! I've done a thorough analysis for you. My AI system predicts a 92% risk of interactions here, which is quite high. My main concern is that my information shows a serious interaction between Warfarin and Ibuprofen, which can *significantly increase your risk of bleeding*. The dosage you've entered (500mg twice daily) is within safe limits, but the interaction risk is too high. Because of this clear risk, my advice is not to take this combination."
        * *If no interactions are found:* Be reassuring. *Example:* "Hi Ganesh! Great news! I've checked my information thoroughly, and I don't see any major interactions listed for this medication with your current regimen. My AI analysis shows only a 5% risk of interactions, which is very low. The dosage you've entered also looks safe. This medication appears to be a good fit for you!"
    6.  *Final Verdict:* End your response with a clear, one-line verdict on a new line: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD".
    """
    
    # Try cloud API first (OpenRouter.ai), fallback to localhost
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    
    if api_key:
        # Use OpenRouter.ai cloud API
        print("[INFO] Using OpenRouter.ai cloud API")
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Medicine Assistant"
        }
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 1500
        }
    else:
        # Fallback to local API (original behavior)
        print("[INFO] Using local LLM API")
        api_url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM API error: {e}")
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
def get_holistic_context(new_drug, patient, existing_meds, dosage_amount=None, dosage_unit=None, frequency=None):
    patient_age = calculate_age(patient.get('dob'))
    patient['age'] = patient_age
    
    context_str = f"Patient Profile: Name: {patient.get('name')}, Age: {patient_age}, Conditions: {patient.get('conditions')}, Allergies: {patient.get('drug_allergies')}, Smoker: {patient.get('is_smoker')}, Alcohol: {patient.get('alcohol_consumption')}.\n"
    context_str += f"Current Medications: {', '.join([med['drug_name'] for med in existing_meds]) if existing_meds else 'None'}.\n"
    context_str += f"New Drug to Analyze: {new_drug}.\n\n"
    
    # Add GNN prediction if model is available
    if gnn_model and drug_map and existing_meds:
        gnn_risk = get_gnn_prediction(new_drug, existing_meds)
        context_str += f"GNN Predicted Risk: {gnn_risk:.1f}% chance of interaction\n\n"
    else:
        context_str += "GNN Predicted Risk: Unable to calculate (model not available or no existing medications)\n\n"
    
    # Add side effects information
    side_effects_info = side_effects_db.get_side_effects(new_drug, patient)
    if side_effects_info['side_effects']:
        context_str += f"Common Side Effects of {new_drug}:\n"
        for effect in side_effects_info['side_effects'][:3]:  # Limit to 3 most common
            context_str += f"- {effect}\n"
        context_str += "\n"
    
    if side_effects_info['risk_warnings']:
        context_str += f"Personalized Risk Warnings for {patient.get('name')}:\n"
        for warning in side_effects_info['risk_warnings']:
            context_str += f"- {warning}\n"
        context_str += "\n"
    
    # Add multi-drug conflict analysis
    triangular_conflicts = multi_drug_checker.check_triangular_conflicts(new_drug, existing_meds)
    category_conflicts = multi_drug_checker.check_category_conflicts(new_drug, existing_meds)
    
    if triangular_conflicts:
        context_str += "Multi-Drug Conflict Analysis (Triangular Conflicts):\n"
        for conflict in triangular_conflicts:
            context_str += f"- {conflict['warning']}\n"
            context_str += f"  Drugs involved: {', '.join(conflict['drugs'])}\n"
            context_str += f"  Severity: {conflict['severity']}\n"
        context_str += "\n"
    
    if category_conflicts:
        context_str += "Category-Based Conflict Analysis:\n"
        for conflict in category_conflicts:
            context_str += f"- {conflict['warning']}\n"
        context_str += "\n"
    
    # Add personalized dosage analysis
    if dosage_amount:
        dosage_analysis = calculate_personalized_dosage(new_drug, dosage_amount, patient)
        context_str += f"Personalized Dosage Analysis:\n"
        context_str += f"- Original dose: {dosage_analysis['original_dose']} mg\n"
        context_str += f"- Suggested dose: {dosage_analysis['suggested_dose']} mg\n"
        context_str += f"- Adjustment factor: {dosage_analysis['adjustment_factor']}\n"
        if dosage_analysis['adjustment_reasons']:
            context_str += f"- Adjustment reasons: {', '.join(dosage_analysis['adjustment_reasons'])}\n"
        context_str += "\n"
    
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

def get_gnn_prediction(new_drug, existing_meds):
    """Get GNN prediction for drug interactions"""
    try:
        if not gnn_model or not drug_map:
            return 0.0
        
        # Convert drug names to indices
        new_drug_idx = None
        existing_drug_indices = []
        
        # Find new drug index
        for drug_name, idx in drug_map.items():
            if new_drug.lower() in drug_name.lower() or drug_name.lower() in new_drug.lower():
                new_drug_idx = idx
                break
        
        if new_drug_idx is None:
            return 0.0
        
        # Find existing drug indices
        for med in existing_meds:
            for drug_name, idx in drug_map.items():
                if med['drug_name'].lower() in drug_name.lower() or drug_name.lower() in med['drug_name'].lower():
                    existing_drug_indices.append(idx)
                    break
        
        if not existing_drug_indices:
            return 0.0
        
        # Calculate average risk across all existing medications
        total_risk = 0.0
        count = 0
        
        for existing_idx in existing_drug_indices:
            # Create edge index for this pair
            edge_index = torch.tensor([[new_drug_idx], [existing_idx]], dtype=torch.long)
            
            # Get node embeddings
            with torch.no_grad():
                # Create a simple edge index for the model
                all_nodes = torch.tensor(list(range(len(drug_map))), dtype=torch.long)
                edge_index_full = torch.tensor([[new_drug_idx, existing_idx], [existing_idx, new_drug_idx]], dtype=torch.long)
                
                # Get predictions
                z = gnn_model.encode(all_nodes, edge_index_full)
                pred = gnn_model.decode(z, edge_index)
                risk_score = torch.sigmoid(pred).item()
                
                total_risk += risk_score
                count += 1
        
        if count > 0:
            avg_risk = (total_risk / count) * 100
            return min(avg_risk, 100.0)  # Cap at 100%
        
        return 0.0
        
    except Exception as e:
        print(f"Error in GNN prediction: {e}")
        return 0.0

@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    try:
        if 'patient_id' not in session: 
            return jsonify({'error': 'User not logged in'}), 401
        
        data = request.json
        if not data or 'drug_name' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        new_drug = data['drug_name']
        dosage_amount = data.get('dosage_amount', '')
        dosage_unit = data.get('dosage_unit', '')
        frequency = data.get('frequency', '')
        
        # Validate dosage first
        dosage_validation = dosage_validator.validate_dosage(new_drug, dosage_amount, dosage_unit, frequency)
        
        conn = get_db_connection()
        patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        if not patient_data:
            conn.close()
            return jsonify({'error': 'Patient not found'}), 404
            
        patient = dict(patient_data)
        existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
        conn.close()
        
        # Create enhanced context including all new features
        holistic_context = get_holistic_context(new_drug, patient, existing_meds, dosage_amount, dosage_unit, frequency)
        
        # Add dosage validation results to context
        if dosage_amount and dosage_unit:
            dosage_info = f"\nDosage Information:\n"
            dosage_info += f"Drug: {new_drug}\n"
            dosage_info += f"Amount: {dosage_amount} {dosage_unit}\n"
            dosage_info += f"Frequency: {frequency or 'Not specified'}\n"
            dosage_info += f"Calculated Daily Dosage: {dosage_validation.get('daily_dosage', 'N/A')} {dosage_validation.get('unit', '')}\n"
            
            if dosage_validation['warnings']:
                dosage_info += f"Dosage Warnings:\n"
                for warning in dosage_validation['warnings']:
                    dosage_info += f"- {warning}\n"
            
            if dosage_validation['max_daily']:
                dosage_info += f"Maximum Safe Daily Dose: {dosage_validation['max_daily']} {dosage_validation['unit']}\n"
            if dosage_validation['max_single']:
                dosage_info += f"Maximum Safe Single Dose: {dosage_validation['max_single']} {dosage_validation['unit']}\n"
            
            holistic_context += dosage_info
        
        # Get AI summary with error handling
        try:
            ai_summary = ask_local_llm(holistic_context)
            summary_lines = ai_summary.split('\n')
            verdict = summary_lines[-1] if summary_lines else "Verdict: DO NOT ADD"
        except Exception as e:
            print(f"Error getting AI summary: {e}")
            ai_summary = f"I apologize, but I'm having trouble connecting to the AI assistant right now. Please try again in a moment.\n\nVerdict: DO NOT ADD"
            summary_lines = ai_summary.split('\n')
            verdict = summary_lines[-1]
        
        # Determine if medication can be added based on both interaction and dosage safety
        interaction_safe = "SAFE TO ADD" in verdict
        dosage_safe = dosage_validation['is_safe']
        can_add = interaction_safe and dosage_safe
        
        main_summary = "\n".join(summary_lines[:-1]) if len(summary_lines) > 1 else ai_summary
        
        # Add dosage warnings to the response
        response_data = {
            'summary': main_summary, 
            'can_add': can_add,
            'dosage_validation': dosage_validation
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in check_before_adding: {e}")
        return jsonify({
            'error': 'An error occurred while checking the medication',
            'summary': 'I apologize, but there was an error processing your request. Please try again.',
            'can_add': False,
            'dosage_validation': {'is_safe': False, 'warnings': ['Error occurred during validation']}
        }), 500

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
    """AI Assistant endpoint - now returns JSON for chatbot"""
    if 'patient_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        conn = get_db_connection()
        patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        patient = dict(patient_data)
        existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
        conn.close()
        
        # Extract drug name if asking about a specific drug
        new_drug_match = re.search(r'(take|about|check|add)\s+([\w\s-]+)\??', question.lower())
        topic_to_check = new_drug_match.group(2).strip() if new_drug_match else question
        
        # Build context with GNN prediction
        holistic_context = get_holistic_context(topic_to_check, patient, existing_meds)
        
        # Get AI response
        ai_response = ask_local_llm(holistic_context)
        
        # Extract verdict if present
        response_parts = ai_response.split('\nVerdict:')
        main_response = response_parts[0].strip()
        verdict = response_parts[1].strip() if len(response_parts) > 1 else None
        
        return jsonify({
            'response': main_response,
            'verdict': verdict,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] AI Assistant error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process question'}), 500

@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- Live Health Monitoring Endpoints ---
@app.route('/api/health-data')
def get_health_data():
    """Get live health data (simulated for demo)"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # Simulate live health data
    import random
    from datetime import datetime, timedelta
    
    # Generate realistic health data
    heart_rate = random.randint(65, 85)
    steps = random.randint(8000, 12000)
    calories = random.randint(200, 400)
    
    # Generate 7-day trend data
    trend_data = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        trend_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'heart_rate': random.randint(70, 80),
            'steps': random.randint(7000, 11000),
            'calories': random.randint(180, 350)
        })
    
    return jsonify({
        'current': {
            'heart_rate': heart_rate,
            'steps': steps,
            'calories': calories,
            'timestamp': datetime.now().isoformat()
        },
        'trends': trend_data,
        'status': 'connected'
    })

@app.route('/api/google-fit-auth')
def google_fit_auth():
    """Initiate Google Fit OAuth flow"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # In a real implementation, this would redirect to Google OAuth
    # For demo purposes, return instructions
    return jsonify({
        'message': 'Google Fit integration requires OAuth setup',
        'instructions': [
            '1. Install Google Fit app on your phone',
            '2. Enable permissions for step counting and heart rate',
            '3. The app will automatically sync data',
            '4. Currently showing simulated data for demo'
        ],
        'demo_mode': True
    })

@app.route('/api/health-alerts')
def get_health_alerts():
    """Get health alerts based on current data"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    conn = get_db_connection()
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    
    alerts = []
    
    # Check for medication-related alerts
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    
    # Example alerts based on patient profile
    if patient_data['is_smoker'] == 'Yes':
        alerts.append({
            'type': 'warning',
            'title': 'Smoking Alert',
            'message': 'Smoking can affect medication effectiveness. Consider discussing with your doctor.',
            'icon': '🚭'
        })
    
    if patient_data['weight_kg'] and float(patient_data['weight_kg']) > 100:
        alerts.append({
            'type': 'info',
            'title': 'Weight Consideration',
            'message': 'Your weight may affect medication dosing. Monitor for any unusual effects.',
            'icon': '⚖️'
        })
    
    return jsonify({
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    })

# --- Emergency Mode ---
@app.route('/emergency-check', methods=['POST'])
def emergency_check():
    """Quick drug interaction check with GNN prediction and detailed cause"""
    try:
        print(f"[DEBUG] Emergency check received: {request.json}")
        
        data = request.json
        if not data:
            return jsonify({'status': 'UNSAFE', 'reason': 'No data provided', 'error': 'Invalid request'}), 400
        
        if 'drug1' not in data or 'drug2' not in data:
            return jsonify({'status': 'UNSAFE', 'reason': 'Please provide two drug names', 'error': 'Missing drug names'}), 400
        
        drug1 = data['drug1'].strip()
        drug2 = data['drug2'].strip()
        
        if not drug1 or not drug2:
            return jsonify({'status': 'UNSAFE', 'reason': 'Both drug names are required', 'error': 'Empty drug names'}), 400
        
        print(f"[DEBUG] Checking interaction between: {drug1} and {drug2}")
        
        # Get GNN prediction
        gnn_risk = 0.0
        if gnn_model and drug_map:
            try:
                # Create a mock medication list for GNN prediction
                mock_meds = [{'drug_name': drug2}]
                gnn_risk = get_gnn_prediction(drug1, mock_meds)
                print(f"[DEBUG] GNN Risk: {gnn_risk}%")
            except Exception as e:
                print(f"[ERROR] GNN prediction failed: {e}")
        
        # Quick interaction check from RAG
        interaction = rag_system.search_interaction(drug1, drug2)
        print(f"[DEBUG] Interaction found: {interaction}")
        
        # Build detailed response
        response_text = ""
        status = "SAFE"
        
        # Add GNN prediction
        if gnn_risk > 0:
            response_text += f"🤖 **AI Prediction:** {gnn_risk:.1f}% interaction risk\n\n"
        
        # Determine safety based on interaction
        if interaction:
            severity = interaction.get('severity', 'Unknown')
            interaction_desc = interaction.get('interaction', 'Unknown interaction')
            
            # Check severity to determine safety
            if severity.lower() in ['major', 'severe', 'contraindicated']:
                status = 'UNSAFE'
                response_text += f"⚠️ **HIGH RISK:** {severity.upper()} interaction detected\n\n"
                response_text += f"**What happens:** {interaction_desc}\n\n"
                response_text += f"**Why this is dangerous:** This combination can cause serious health complications. "
                response_text += f"The drugs interact in a way that may increase side effects or reduce effectiveness.\n\n"
                response_text += f"**Recommendation:** DO NOT take these together without doctor supervision."
            elif severity.lower() in ['moderate', 'moderate risk']:
                status = 'CAUTION'
                response_text += f"⚠️ **MODERATE RISK:** {severity.upper()} interaction detected\n\n"
                response_text += f"**What happens:** {interaction_desc}\n\n"
                response_text += f"**Why you should be careful:** This combination may cause unwanted effects or reduce drug effectiveness.\n\n"
                response_text += f"**Recommendation:** Consult your doctor before taking these together."
            else:
                status = 'SAFE'
                response_text += f"ℹ️ **Minor interaction:** {interaction_desc}\n\n"
                response_text += f"**Recommendation:** Generally safe, but monitor for any unusual effects."
        else:
            if gnn_risk > 70:
                status = 'CAUTION'
                response_text += f"⚠️ **AI detected potential risk** even though no documented interaction was found.\n\n"
                response_text += f"**Recommendation:** Consult a healthcare professional to be safe."
            else:
                status = 'SAFE'
                response_text += f"✅ **{drug1} and {drug2} appear safe to take together**\n\n"
                response_text += f"No known interactions found in our database.\n\n"
                response_text += f"**Note:** Always inform your doctor about all medications you're taking."
        
        print(f"[DEBUG] Status: {status}, GNN Risk: {gnn_risk}%")
        
        return jsonify({
            'status': status,
            'response': response_text,
            'gnn_risk': round(gnn_risk, 1),
            'drug1': drug1,
            'drug2': drug2,
            'interaction': interaction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Exception in emergency check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'UNSAFE',
            'response': 'Unable to perform emergency check. Please consult a healthcare professional immediately.',
            'error': str(e)
        }), 500

# --- Google Fit OAuth2 Integration ---
# Configuration (you'll need to set these as environment variables)
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:5000/oauth2callback')

SCOPES = ['https://www.googleapis.com/auth/fitness.heart_rate.read',
          'https://www.googleapis.com/auth/fitness.activity.read',
          'https://www.googleapis.com/auth/fitness.body.read']

def credentials_to_dict(credentials):
    """Convert credentials object to dictionary for storage in session"""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@app.route('/authorize-fit')
def authorize_fit():
    """Initiate Google Fit OAuth2 flow"""
    print("[DEBUG] Starting Google Fit authorization")
    
    if not GOOGLE_FIT_AVAILABLE:
        return jsonify({
            'error': 'Google Fit not available',
            'message': 'Please install: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client'
        }), 500
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({
            'error': 'Google Fit not configured',
            'message': 'Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables'
        }), 500
    
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [GOOGLE_REDIRECT_URI]
                }
            },
            scopes=SCOPES
        )
        
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        session['state'] = state
        print(f"[DEBUG] Redirecting to: {authorization_url}")
        
        return redirect(authorization_url)
        
    except Exception as e:
        print(f"[ERROR] Authorization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/oauth2callback')
def oauth2callback():
    """Handle Google OAuth2 callback"""
    print("[DEBUG] OAuth2 callback received")
    
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [GOOGLE_REDIRECT_URI]
                }
            },
            scopes=SCOPES,
            state=session['state']
        )
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        
        authorization_response = request.url
        flow.fetch_token(authorization_response=authorization_response)
        
        # Store credentials in session
        session['credentials'] = credentials_to_dict(flow.credentials)
        print("[DEBUG] Credentials stored in session")
        
        flash('Successfully connected to Google Fit!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"[ERROR] Callback error: {e}")
        flash('Failed to connect to Google Fit. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/disconnect-fit')
def disconnect_fit():
    """Disconnect Google Fit"""
    session.pop('credentials', None)
    flash('Disconnected from Google Fit', 'info')
    return redirect(url_for('dashboard'))

@app.route('/get-fit-data')
def get_fit_data():
    """Fetch real-time health data from Google Fit API"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # Check if Google Fit is connected
    if 'credentials' not in session:
        # Return fallback data if not connected
        return get_dummy_health_data()
    
    try:
        # Get credentials from session
        creds_dict = session['credentials']
        
        # Build credentials object
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=creds_dict.get('token'),
            refresh_token=creds_dict.get('refresh_token'),
            token_uri=creds_dict.get('token_uri'),
            client_id=creds_dict.get('client_id'),
            client_secret=creds_dict.get('client_secret'),
            scopes=creds_dict.get('scopes')
        )
        
        # Refresh if necessary
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            session['credentials'] = credentials_to_dict(creds)
        
        # Fetch health data
        health_data = fetch_google_fit_data(creds)
        
        return jsonify({
            'status': 'connected',
            'data_source': 'google_fit',
            'current': health_data['current'],
            'trends': health_data['trends'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Error fetching Google Fit data: {e}")
        return get_dummy_health_data()

def fetch_google_fit_data(credentials):
    """Fetch data from Google Fit API"""
    try:
        import googleapiclient.discovery
        
        fit_service = googleapiclient.discovery.build('fitness', 'v1', credentials=credentials)
        
        # Define time range (last 24 hours)
        end_time = int(datetime.now().timestamp() * 1000000000)  # nanoseconds
        start_time = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000000000)
        
        # Get heart rate
        heart_rate_data = fit_service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm',
            datasetId=f'{start_time}-{end_time}'
        ).execute()
        
        # Get steps
        steps_data = fit_service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.step_count.delta:com.google.android.gms:estimated_steps',
            datasetId=f'{start_time}-{end_time}'
        ).execute()
        
        # Process heart rate
        heart_rate = 75  # default
        if 'point' in heart_rate_data and len(heart_rate_data['point']) > 0:
            latest_point = heart_rate_data['point'][-1]
            heart_rate = int(latest_point['value'][0]['fpVal'])
        
        # Process steps
        steps = 0
        if 'point' in steps_data:
            for point in steps_data['point']:
                steps += int(point['value'][0]['intVal'])
        
        # Get calories (estimated)
        calories = int(steps * 0.05)  # Rough estimate: 0.05 cal per step
        
        # Generate trend data (last 7 days)
        trends = []
        for i in range(7):
            trends.append({
                'date': (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d'),
                'heart_rate': heart_rate + random.randint(-5, 5),
                'steps': max(0, steps + random.randint(-1000, 1000)),
                'calories': int((steps + random.randint(-1000, 1000)) * 0.05)
            })
        
        return {
            'current': {
                'heart_rate': heart_rate,
                'steps': steps,
                'calories': calories
            },
            'trends': trends
        }
        
    except Exception as e:
        print(f"[ERROR] Error in fetch_google_fit_data: {e}")
        return get_dummy_health_data()

def get_dummy_health_data():
    """Generate dummy health data as fallback"""
    heart_rate = random.randint(65, 85)
    steps = random.randint(8000, 12000)
    calories = random.randint(200, 400)
    
    trends = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        trends.append({
            'date': date.strftime('%Y-%m-%d'),
            'heart_rate': random.randint(70, 80),
            'steps': random.randint(7000, 11000),
            'calories': random.randint(180, 350)
        })
    
    return jsonify({
        'status': 'simulated',
        'data_source': 'dummy',
        'current': {
            'heart_rate': heart_rate,
            'steps': steps,
            'calories': calories,
            'timestamp': datetime.now().isoformat()
        },
        'trends': trends
    })

@app.route('/api/google-fit-connect')
def google_fit_connect():
    """Initiate Google Fit connection"""
    return redirect(url_for('authorize_fit'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

