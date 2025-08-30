import sqlite3

# Connect to the database file (it will be created if it doesn't exist)
conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for the new application structure...")

# --- IMPORTANT CHANGE ---
# Drop existing tables first to ensure a clean slate.
# This prevents errors if you run the script multiple times or have an old DB structure.
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables, if any, have been removed.")

# Create the new 'patients' table with the detailed profile schema
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    age INTEGER,
    gender TEXT,
    weight_kg REAL,
    conditions TEXT,
    allergies TEXT
);
''')
print("Table 'patients' created successfully.")

# Create the 'medications' table
cursor.execute('''
CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    drug_name TEXT NOT NULL,
    dosage TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
''')
print("Table 'medications' created successfully.")

conn.commit()
conn.close()

print("\nDatabase setup complete. The 'medicine_log.db' file is fresh and ready.")

