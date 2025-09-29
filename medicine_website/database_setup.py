import sqlite3

# --- This script should be run ONCE to set up your database file ---

conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for the final application structure...")

# Drop existing tables to ensure a clean slate
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables, if any, have been removed.")

# Create the 'patients' table with the comprehensive profile schema
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    
    -- Personal Details
    dob TEXT,
    gender TEXT,
    weight_kg REAL,
    height_cm REAL,
    emergency_contact TEXT,

    -- Medical History & Conditions
    conditions TEXT,
    
    -- Allergies
    drug_allergies TEXT,
    food_allergies TEXT,
    other_allergies TEXT,

    -- Lifestyle
    is_smoker TEXT,
    alcohol_consumption TEXT
);
''')
print("Table 'patients' created successfully.")

# Create the 'medications' table with detailed dosage fields
cursor.execute('''
CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    drug_name TEXT NOT NULL,
    dosage_amount REAL,
    dosage_unit TEXT,
    frequency TEXT,
    start_date TEXT,
    end_date TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
''')
print("Table 'medications' created successfully.")

conn.commit()
conn.close()

print("\nDatabase setup complete. The 'medicine_log.db' file is fresh and ready.")

