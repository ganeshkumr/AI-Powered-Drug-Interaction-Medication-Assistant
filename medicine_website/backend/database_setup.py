import sqlite3

# --- This script should be run ONCE to set up your database file ---

conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for secure email-based authentication...")

# Drop existing tables to ensure a clean slate with the new structure
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables dropped.")

# Create the 'patients' table with a new 'email' column for login
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE, -- Email is now the unique identifier for login
    password_hash TEXT NOT NULL,
    name TEXT, -- Name is now just a display field, not for login
    
    -- All other profile fields remain the same
    dob TEXT,
    gender TEXT,
    weight_kg REAL,
    height_cm REAL,
    emergency_contact TEXT,
    conditions TEXT,
    drug_allergies TEXT,
    food_allergies TEXT,
    other_allergies TEXT,
    is_smoker TEXT,
    alcohol_consumption TEXT
);
''')
print("Table 'patients' created with secure email and password fields.")

# Recreate the 'medications' table
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
print("Table 'medications' created.")

conn.commit()
conn.close()

print("\nDatabase setup complete. The database is now ready for secure, email-based logins.")

