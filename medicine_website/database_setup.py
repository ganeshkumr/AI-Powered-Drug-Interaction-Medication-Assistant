import sqlite3

conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for detailed medication logging...")

# Drop existing tables to ensure a clean slate with the new structure
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables dropped.")

# Recreate the 'patients' table
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
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
print("Table 'patients' created.")

# Create the new 'medications' table with a much more detailed schema
cursor.execute('''
CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    drug_name TEXT NOT NULL,
    dosage_amount REAL,
    dosage_unit TEXT, -- e.g., 'mg', 'ml', 'tablet(s)'
    frequency TEXT, -- e.g., 'Once daily', 'Twice daily'
    start_date TEXT,
    end_date TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
''')
print("Table 'medications' created with new detailed dosage fields.")

conn.commit()
conn.close()

print("\nDatabase setup complete. The database is now ready for the new medication tracking features.")

