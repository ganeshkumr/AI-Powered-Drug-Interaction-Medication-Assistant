import sqlite3

# Connect to the database file (it will be created if it doesn't exist)
conn = sqlite3.connect('medicine_log.db')
cursor = conn.cursor()

print("Database connected. Creating/updating tables...")

# Drop existing tables to ensure a clean slate with the new structure
cursor.execute('DROP TABLE IF EXISTS medications;')
cursor.execute('DROP TABLE IF EXISTS patients;')
print("Old tables dropped.")

# Create the new 'patients' table with a more detailed profile
# We are adding name, age, gender, weight, conditions, and allergies.
cursor.execute('''
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, -- Name will also be our simple login username
    age INTEGER,
    gender TEXT,
    weight_kg REAL,
    conditions TEXT,
    allergies TEXT
);
''')
print("Table 'patients' created with new schema.")

# Create the 'medications' table
# 'patient_id' is a foreign key that links this table to the 'patients' table.
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

# Commit the changes and close the connection
conn.commit()
conn.close()

print("\nDatabase setup complete. The 'medicine_log.db' file is ready.")
print("You only need to run this script once.")

