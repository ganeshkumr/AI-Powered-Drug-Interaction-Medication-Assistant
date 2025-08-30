    # Fetch all patients
from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Helper function to get a database connection
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db')
    # This allows us to access columns by name (like a dictionary)
    conn.row_factory = sqlite3.Row
    return conn

# Main route: Display all patients and their medications
@app.route('/')
def index():
    conn = get_db_connection()
    
    # Fetch all patients
    patients_data = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    
    # For each patient, fetch their medications
    patients = []
    for p_data in patients_data:
        patient = dict(p_data) # Convert the Row object to a dictionary
        medications = conn.execute(
            'SELECT * FROM medications WHERE patient_id = ? ORDER BY drug_name', 
            (patient['id'],)
        ).fetchall()
        patient['medications'] = medications
        patients.append(patient)
        
    conn.close()
    # Pass the complete data to the HTML template
    return render_template('index.html', patients=patients)

# Route to handle adding a new patient
@app.route('/add_patient', methods=['POST'])
def add_patient():
    # Get data from the form submitted by the user
    name = request.form['name']
    age = request.form['age']
    conditions = request.form['conditions']
    allergies = request.form['allergies']

    conn = get_db_connection()
    conn.execute(
        'INSERT INTO patients (name, age, conditions, allergies) VALUES (?, ?, ?, ?)',
        (name, age, conditions, allergies)
    )
    conn.commit()
    conn.close()
    
    # Redirect back to the main page to see the new patient
    return redirect(url_for('index'))

# Route to handle adding a new medication for a specific patient
@app.route('/add_medication', methods=['POST'])
def add_medication():
    patient_id = request.form['patient_id']
    drug_name = request.form['drug_name']
    dosage = request.form['dosage']

    conn = get_db_connection()
    conn.execute(
        'INSERT INTO medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)',
        (patient_id, drug_name, dosage)
    )
    conn.commit()
    conn.close()
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(debug=True)
