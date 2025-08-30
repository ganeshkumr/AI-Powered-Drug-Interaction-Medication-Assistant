import sqlite3

# Connect to the database file (it will be created if it doesn't exist)
conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Creating tables...")

# Create the 'patients' table
# id is the primary key, which uniquely identifies each patient.
cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    conditions TEXT,
    allergies TEXT
);
''')
print("Table 'patients' created or already exists.")

# Create the 'medications' table
# 'patient_id' is a foreign key that links this table to the 'patients' table.
cursor.execute('''
CREATE TABLE IF NOT EXISTS medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    drug_name TEXT NOT NULL,
    dosage TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
''')
print("Table 'medications' created or already exists.")

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database setup complete. The 'medicine_log.db' file is ready.")
