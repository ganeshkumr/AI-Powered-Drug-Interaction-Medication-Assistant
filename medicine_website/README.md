# Medicine Safety Assistant

A full-stack medication safety platform that combines classical interaction data, model-driven risk scoring, and profile-aware guidance.

## Overview

Medicine Safety Assistant helps users:
- check medicine interaction risk before adding a drug,
- maintain a personal medication list,
- review profile-based risk context,
- run quick anonymous checks without login.

The project uses a Flask backend for clinical logic + data services and a React/Vite frontend for the user experience.

## Architecture

### Backend (Flask)
- Main runtime: `backend/app.py`
- Data store: SQLite (`backend/medicine_log.db`)
- Core data files: `backend/data/*`
- Model artifacts: `backend/models/*`

### Frontend (React + Vite)
- App root: `medicine-assistant-react/`
- Runtime UI source: `medicine-assistant-react/src/`
- API base expected by frontend: `http://127.0.0.1:5000`

## Repository Structure

```text
medicine_website/
|-- backend/
|   |-- app.py
|   |-- medicine_log.db
|   |-- data/
|   |   |-- interactions.csv
|   |   |-- drugbank_drug_list.json
|   |   |-- side_effects_database_drugbank.json
|   |   |-- drug_dosage_limits_drugbank.json
|   |   `-- multi_drug_conflicts.json
|   |-- models/
|   |   |-- gnn_model.pt
|   |   `-- drug_map.json
|   `-- README.md
|-- medicine-assistant-react/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
|-- data/                  # optional/auxiliary raw datasets
|-- DRUG_bank dataset/     # source dataset assets
|-- run_backend.bat
`-- README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Quick Start

### 1. Start backend

From project root:

```powershell
run_backend.bat
```

or manually:

```powershell
cd backend
python app.py
```

Backend URL: `http://127.0.0.1:5000`

### 2. Start frontend

```powershell
cd medicine-assistant-react
npm.cmd install
npm.cmd run dev
```

Frontend URL: `http://127.0.0.1:5173`

## Environment Variables

Use `.env` in project root (already present in your setup):

```env
OPENROUTER_API_KEY=your_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

## API Surface (Active)

### Auth
- `GET /api/check-auth`
- `POST /api/login`
- `POST /api/register`
- `GET /api/logout`

### Profile & Medications
- `GET /api/profile`
- `POST /api/profile`
- `GET /api/medications`
- `DELETE /api/medications/<id>`
- `POST /add_medication`

### Safety / Analysis
- `POST /api/quick-check`
- `POST /check_before_adding`
- `GET /api/search-drugs`

### Health Integrations
- `GET /api/health-data`
- `GET /api/google-fit-auth`
- `GET /api/health-alerts`

## Data and Model Notes

- Active runtime data/model files are under `backend/data` and `backend/models`.
- Root-level duplicate model/app files were removed to avoid confusion.
- `data/` at repo root can be kept for raw/processing artifacts, but runtime depends on `backend/data`.

## Operational Checks

### Backend health check

```powershell
Invoke-WebRequest http://127.0.0.1:5000/api/check-auth -UseBasicParsing
```

Expected: HTTP `200` with JSON like `{"authenticated": false}` when not logged in.

### Frontend production build

```powershell
cd medicine-assistant-react
npm.cmd run build
```

## Troubleshooting

### PowerShell blocks `npm` (`npm.ps1 cannot be loaded`)
Use `npm.cmd` instead of `npm`:

```powershell
npm.cmd run dev
npm.cmd run build
```

### Frontend `spawn EPERM` (sandbox/permission constraints)
Run commands in normal terminal with permissions, or elevated if required by environment policy.

### Login/session issues
- Ensure backend runs at `127.0.0.1:5000`.
- Frontend must use `withCredentials: true` (already configured in `src/services/api.js`).

## Safety Disclaimer

This software is for informational support only and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medication decisions.
