# Backend - Medicine Safety Assistant

This folder contains all backend files for the Medicine Safety Assistant application.

## Structure

```
backend/
├── app.py                          # Main Flask application
├── database_setup.py               # Database initialization
├── response_validation.py          # Response validation utilities
├── extract_drugbank_complete.py    # DrugBank data extraction tool
├── train_gnn.py                    # GNN model training script
├── process_kaggle_data.py          # Data processing utilities
├── medicine_log.db                 # SQLite database
├── templates/                      # HTML templates for Flask
├── data/                          # All data files
│   ├── interactions.csv           # Drug-drug interactions (main database)
│   ├── drug_dosage_limits_drugbank.json  # DrugBank dosage data (15 drugs)
│   ├── side_effects_database_drugbank.json  # DrugBank side effects (47,711 drugs)
│   ├── drugbank_drug_list.json    # Complete list of 73,687 drugs
│   ├── multi_drug_conflicts.json  # Multi-drug conflict patterns
│   └── drug_conditions.json       # Drug-condition relationships
└── models/                        # GNN model files
    ├── gnn_model.pt              # Trained GNN model
    └── drug_map.json             # Drug name to index mapping
```

## Key Features

### 1. Comprehensive Drug Database (100% DrugBank)
- **73,687 drugs** from DrugBank XML database
- **47,711 side effects entries** with comprehensive coverage including synonyms
- **Drug-drug interactions** from validated medical sources (Kaggle dataset)
- **Dosage limits** for 15 drugs extracted from DrugBank (dosage data is sparse in DrugBank)

### 2. AI-Powered Analysis
- **GNN (Graph Neural Network)** for interaction risk prediction
- **RAG (Retrieval-Augmented Generation)** for factual interaction lookup
- **LLM integration** for personalized explanations

### 3. Safety Features
- Dosage validation against maximum safe limits
- Multi-drug conflict detection
- Patient-specific risk assessment
- Allergy checking

## Running the Backend

### Prerequisites
```bash
pip install flask flask-cors pandas torch torch-geometric requests python-dotenv
```

### Start the Server
```bash
cd backend
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - User login
- `POST /api/logout` - User logout

### Medication Management
- `GET /api/medications` - Get user's medications
- `POST /api/medications` - Add medication
- `DELETE /api/medications/<id>` - Delete medication

### Safety Analysis
- `POST /api/check-before-adding` - Check drug safety before adding
- `POST /api/quick-check` - Quick safety check (no login required)

### User Profile
- `GET /api/profile` - Get user profile
- `PUT /api/profile` - Update user profile

## Data Sources

1. **DrugBank** - Comprehensive drug database
   - 73,687 drugs with detailed information
   - Side effects, interactions, and pharmacology

2. **Kaggle Drug Interactions** - Validated interaction data
   - Pairwise drug interactions
   - Severity classifications

3. **GNN Model** - Machine learning predictions
   - Trained on drug interaction patterns
   - Provides risk probability scores

## Environment Variables

Create a `.env` file in the backend directory:

```env
OPENROUTER_API_KEY=your_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

## Database Schema

### Users Table
- id, name, email, password, dob, gender, conditions, allergies, etc.

### Medications Table
- id, user_id, drug_name, dosage_amount, dosage_unit, frequency, added_date

## Notes

- The database file `medicine_log.db` is created automatically on first run
- DrugBank data provides the most comprehensive coverage
- GNN model requires training before first use (run `train_gnn.py`)
- All file paths are relative to the backend directory

## Updating DrugBank Data

To extract fresh data from DrugBank:

```bash
python extract_drugbank_complete.py
```

This will regenerate:
- `data/drug_dosage_limits_drugbank.json`
- `data/side_effects_database_drugbank.json`
- `data/drugbank_drug_list.json`
