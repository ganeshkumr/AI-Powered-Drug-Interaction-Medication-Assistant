import sqlite3

conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Resetting tables for secure authentication...")

# Drop existing tables to ensure a clean slate with the new password column
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables dropped.")

# Create the 'patients' table with a new 'password_hash' column
# This column will store the secure hash, not the plain-text password.
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    weight_kg REAL,
    conditions TEXT,
    allergies TEXT
);
''')
print("Table 'patients' created with secure password field.")

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

print("\nDatabase setup complete. The database is now ready for secure logins.")

