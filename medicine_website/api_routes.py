# API Routes for React Frontend
# Add these routes to your app.py file before "if __name__ == '__main__':"

from flask import jsonify
from flask_cors import CORS

# Enable CORS for React (add this near the top of app.py after app = Flask(__name__))
# CORS(app, supports_credentials=True)

# API: Check authentication status
@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'patient_id' in session:
        conn = get_db_connection()
        patient = conn.execute('SELECT id, name, email FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        conn.close()
        if patient:
            return jsonify({
                'authenticated': True,
                'user': {'id': patient['id'], 'name': patient['name'], 'email': patient['email']}
            })
    return jsonify({'authenticated': False})

# API: Login endpoint for React
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if patient and check_password_hash(patient['password_hash'], password):
        session['patient_id'] = patient['id']
        return jsonify({
            'success': True,
            'user': {'id': patient['id'], 'name': patient['name'], 'email': patient['email']}
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

# API: Register endpoint for React
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not is_valid_email(email):
        return jsonify({'success': False, 'error': 'Invalid email address'}), 400
    
    is_strong, message = is_strong_password(password)
    if not is_strong:
        return jsonify({'success': False, 'error': message}), 400
    
    conn = get_db_connection()
    if conn.execute('SELECT id FROM patients WHERE email = ?', (email,)).fetchone():
        conn.close()
        return jsonify({'success': False, 'error': 'Email already exists'}), 400
    
    password_hash = generate_password_hash(password)
    conn.execute('INSERT INTO patients (name, email, password_hash) VALUES (?, ?, ?)', (name, email, password_hash))
    conn.commit()
    new_patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    session['patient_id'] = new_patient['id']
    return jsonify({
        'success': True,
        'user': {'id': new_patient['id'], 'name': new_patient['name'], 'email': new_patient['email']}
    })

# API: Logout
@app.route('/api/logout', methods=['GET'])
def api_logout():
    session.pop('patient_id', None)
    return jsonify({'success': True})

# API: Get health data
@app.route('/api/health-data', methods=['GET'])
def api_health_data():
    if 'patient_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Mock health data - replace with real Google Fit data
    current_data = {
        'heart_rate': random.randint(60, 100),
        'steps': random.randint(3000, 15000),
        'calories': random.randint(1500, 3000)
    }
    
    # Generate 7 days of trend data
    trends = []
    for i in range(7):
        day = datetime.now() - timedelta(days=6-i)
        trends.append({
            'date': day.isoformat(),
            'heart_rate': random.randint(60, 100),
            'steps': random.randint(3000, 15000),
            'calories': random.randint(1500, 3000)
        })
    
    return jsonify({'current': current_data, 'trends': trends})

# API: Get medications
@app.route('/api/medications', methods=['GET'])
def api_medications():
    if 'patient_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = get_db_connection()
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()
    
    meds_list = [dict(med) for med in medications]
    return jsonify({'medications': meds_list})
