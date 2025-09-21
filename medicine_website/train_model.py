import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    """Trains the XGBoost model using the real-world dataset and saves it."""
    # 1. Load the Dataset
    try:
        print("Loading the patient dataset...")
        df = pd.read_csv('models/data/patient_ade_dataset.csv')
    except FileNotFoundError:
        print("\n[ERROR] 'data/patient_ade_dataset.csv' not found.")
        print("Please run 'fetch_real_data.py' first to create it.")
        return

    # 2. Define Features and Target
    features = ['age', 'num_conditions', 'num_allergies', 'num_meds']
    target = 'has_ade'
    X = df[features]
    y = df[target]

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training and testing sets.")

    # 4. Train the XGBoost model
    print("Training the XGBoost prediction model...")
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save the trained "brain" to the 'models' folder
    os.makedirs('models', exist_ok=True)
    print("\nSaving the trained model to 'models/ade_predictor.joblib'...")
    joblib.dump(model, 'models/ade_predictor.joblib')
    print("Model saved successfully. It is now ready for the web app.")

if __name__ == '__main__':
    train()

