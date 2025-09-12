import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the trained model

# 1. Load the Dataset
print("Loading the synthetic patient dataset...")
df = pd.read_csv('patient_ade_dataset.csv')

# 2. Define Features (X) and Target (y)
features = ['age', 'num_conditions', 'num_allergies', 'num_meds']
target = 'has_ade'

X = df[features]
y = df[target]

# 3. Split Data into Training and Testing Sets
# We train the model on 80% of the data and test its performance on the unseen 20%.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# 4. Train the Machine Learning Model
print("Training the Random Forest Classifier...")
# We use a RandomForest model, which is excellent for this type of tabular data.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate the Model's Performance
print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Trained Model for the Web App
print("\nSaving the trained model to 'ade_predictor.joblib'...")
joblib.dump(model, 'ade_predictor.joblib')
print("Model saved successfully. It is now ready to be used by the Flask application.")
