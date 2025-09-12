import sqlite3

conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for the new, comprehensive patient profile...")

# Drop existing tables to ensure a clean slate with the new structure
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables dropped.")

# Create the new 'patients' table with a much more detailed schema
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    
    -- Personal Details
    dob TEXT, -- Date of Birth
    gender TEXT,
    weight_kg REAL,
    height_cm REAL,
    emergency_contact TEXT,

    -- Medical History & Conditions
    conditions TEXT, -- For storing multiple, comma-separated values
    
    -- Allergies
    drug_allergies TEXT,
    food_allergies TEXT,
    other_allergies TEXT,

    -- Lifestyle
    is_smoker TEXT, -- e.g., 'Yes', 'No', 'Former'
    alcohol_consumption TEXT -- e.g., 'None', 'Occasional', 'Regular'
);
''')
print("Table 'patients' created with new comprehensive schema.")

# Recreate the 'medications' table
cursor.execute('''
CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    drug_name TEXT NOT NULL,
    dosage TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
''')
print("Table 'medications' created.")

conn.commit()
conn.close()

print("\nDatabase setup complete. The database is now ready for the new detailed profiles.")

