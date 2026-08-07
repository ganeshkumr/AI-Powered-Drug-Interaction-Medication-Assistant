from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date, datetime, timedelta
# PyTorch GNN imports (optional for lightweight deployments)
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"[WARNING] PyTorch/TorchGeometric disabled or not installed: {e}")
import os
import random
import time
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

# Google Fit OAuth2 imports (optional package)
try:
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    GOOGLE_FIT_AVAILABLE = True
except ImportError:
    GOOGLE_FIT_AVAILABLE = False
    print("[WARNING] google-auth-oauthlib not installed. Google Fit features disabled. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'the_final_and_most_secure_key')

# Allow requests from localhost (dev) and the deployed Vercel frontend (prod)
_allowed_origins = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
]
_frontend_url = os.environ.get('FRONTEND_URL', '')
if _frontend_url:
    _allowed_origins.append(_frontend_url)

CORS(app, supports_credentials=True, origins=_allowed_origins)

def load_json_file(file_path, default=None):
    """Load JSON reliably across Windows code pages."""
    if default is None:
        default = {}

    encodings = ("utf-8", "utf-8-sig", "cp1252")
    last_decode_error = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            last_decode_error = e

    if last_decode_error:
        raise last_decode_error
    return default

# --- GNN Model Definition and Loading ---
if TORCH_AVAILABLE:
    class GNNLinkPredictor(torch.nn.Module):
        def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels):
            super(GNNLinkPredictor, self).__init__()
            self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
            self.conv1 = GATConv(embedding_dim, hidden_channels, heads=4, dropout=0.6)
            self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=0.6)

        def encode(self, x, edge_index):
            x = self.embedding(x); x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index)); x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index); return x

        def decode(self, z, edge_label_index):
            src = z[edge_label_index[0]]; dst = z[edge_label_index[1]]; return (src * dst).sum(dim=-1)

def load_gnn_model():
    if not TORCH_AVAILABLE:
        print("[INFO] Running in lightweight RAG mode (PyTorch disabled).")
        return None, None
    try:
        drug_map = load_json_file('models/drug_map.json')
        model = GNNLinkPredictor(num_nodes=len(drug_map), embedding_dim=128, hidden_channels=128, out_channels=128)
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load('models/gnn_model.pt', map_location=map_location))
        model.eval(); print("[INFO] GNN Prediction Model loaded successfully on CPU.")
        return model, drug_map
    except Exception as e:
        print(f"[WARNING] GNN model could not be loaded: {e}")
        return None, None

gnn_model, drug_map = load_gnn_model()

# --- RAG System ---
class RAGSystem:
    def __init__(self, data_file):
        try:
            self.df = pd.read_csv(data_file).fillna('')
            self.df['drug_a_lower'] = self.df['drug_a'].astype(str).str.lower().str.strip()
            self.df['drug_b_lower'] = self.df['drug_b'].astype(str).str.lower().str.strip()
            print("[INFO] RAG Knowledge Base initialized successfully.")
        except Exception as e: print(f"[ERROR] Could not initialize RAG system: {e}"); self.df = None
    def search_interaction(self, drug1, drug2):
        if self.df is None: return None
        d1_lower = drug1.lower().strip(); d2_lower = drug2.lower().strip()
        for _, row in self.df.iterrows():
            row_a = row['drug_a_lower']; row_b = row['drug_b_lower']
            if (d1_lower in row_a and d2_lower in row_b) or \
               (d1_lower in row_b and d2_lower in row_a): return row.to_dict()
        return None
rag_system = RAGSystem('data/interactions.csv')

# --- Drug Dosage Validation System ---
class DosageValidator:
    def __init__(self, dosage_file):
        try:
            self.dosage_limits = load_json_file(dosage_file)
            print("[INFO] Drug dosage limits database initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize dosage validator: {e}")
            self.dosage_limits = {}
    
    def validate_dosage(self, drug_name, dosage_amount, dosage_unit, frequency):
        """Validate drug dosage against maximum safe limits"""
        if not drug_name or not dosage_amount:
            return {"is_safe": True, "warnings": [], "max_daily": None, "max_single": None}
        
        drug_key = drug_name.lower().strip()
        
        # Find matching drug in database
        drug_info = None
        for drug, info in self.dosage_limits.items():
            if drug in drug_key or drug_key in drug:
                drug_info = info
                break
        
        if not drug_info:
            return {"is_safe": True, "warnings": ["No dosage limits found for this medication"], "max_daily": None, "max_single": None}
        
        try:
            dosage_amount = float(dosage_amount)
        except (ValueError, TypeError):
            return {"is_safe": False, "warnings": ["Invalid dosage amount"], "max_daily": None, "max_single": None}
        
        warnings = []
        is_safe = True
        
        # Check unit compatibility
        if dosage_unit and dosage_unit.lower() != drug_info["unit"].lower():
            warnings.append(f"Unit mismatch: Expected {drug_info['unit']}, got {dosage_unit}")
        
        # Calculate daily dosage based on frequency
        daily_dosage = self._calculate_daily_dosage(dosage_amount, frequency)
        
        # Check against maximum limits
        max_daily = drug_info["max_daily_mg"] if "max_daily_mg" in drug_info else drug_info.get("max_daily_iu", drug_info.get("max_daily_units", 0))
        max_single = drug_info["max_single_mg"] if "max_single_mg" in drug_info else drug_info.get("max_single_iu", drug_info.get("max_single_units", 0))
        
        if daily_dosage > max_daily:
            warnings.append(f"Daily dosage ({daily_dosage} {drug_info['unit']}) exceeds maximum safe limit ({max_daily} {drug_info['unit']})")
            is_safe = False
        
        if dosage_amount > max_single:
            warnings.append(f"Single dose ({dosage_amount} {drug_info['unit']}) exceeds maximum safe limit ({max_single} {drug_info['unit']})")
            is_safe = False
        
        # Add general warnings from database
        if drug_info.get("warnings"):
            warnings.extend(drug_info["warnings"])
        
        return {
            "is_safe": is_safe,
            "warnings": warnings,
            "max_daily": max_daily,
            "max_single": max_single,
            "daily_dosage": daily_dosage,
            "unit": drug_info["unit"]
        }
    
    def _calculate_daily_dosage(self, single_dose, frequency):
        """Calculate daily dosage based on frequency"""
        if not frequency:
            return single_dose
        
        frequency_lower = frequency.lower()
        
        # Common frequency patterns
        if any(word in frequency_lower for word in ["once", "daily", "day"]):
            return single_dose
        elif any(word in frequency_lower for word in ["twice", "2x", "two"]):
            return single_dose * 2
        elif any(word in frequency_lower for word in ["three", "3x", "thrice"]):
            return single_dose * 3
        elif any(word in frequency_lower for word in ["four", "4x"]):
            return single_dose * 4
        elif any(word in frequency_lower for word in ["every 6", "6 hours"]):
            return single_dose * 4  # 4 times per day
        elif any(word in frequency_lower for word in ["every 8", "8 hours"]):
            return single_dose * 3  # 3 times per day
        elif any(word in frequency_lower for word in ["every 12", "12 hours"]):
            return single_dose * 2  # 2 times per day
        else:
            # Default to single dose if frequency is unclear
            return single_dose

dosage_validator = DosageValidator('data/drug_dosage_limits_drugbank.json')

# --- Side Effects Database ---
class SideEffectsDatabase:
    def __init__(self, side_effects_file):
        try:
            self.side_effects = load_json_file(side_effects_file)
            print("[INFO] Side effects database initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize side effects database: {e}")
            self.side_effects = {}
    
    def get_side_effects(self, drug_name, patient_profile=None):
        """Get side effects for a drug with personalized risk assessment"""
        drug_key = drug_name.lower().strip()
        
        # Find matching drug
        drug_info = None
        for drug, info in self.side_effects.items():
            if drug in drug_key or drug_key in drug:
                drug_info = info
                break
        
        if not drug_info:
            return {"side_effects": [], "risk_warnings": []}
        
        side_effects = drug_info.get('common_side_effects', [])
        serious_effects = drug_info.get('serious_side_effects', [])
        risk_factors = drug_info.get('risk_factors', {})
        
        # Generate personalized risk warnings
        risk_warnings = []
        if patient_profile:
            for risk_factor, warning in risk_factors.items():
                if self._check_risk_factor(risk_factor, patient_profile):
                    risk_warnings.append(f"⚠️ {warning}")
        
        return {
            "side_effects": side_effects,
            "serious_effects": serious_effects,
            "risk_warnings": risk_warnings
        }
    
    def _check_risk_factor(self, risk_factor, patient_profile):
        """Check if patient has specific risk factors"""
        if risk_factor == "age_65_plus":
            return patient_profile.get('age', 0) >= 65
        elif risk_factor == "age_60_plus":
            return patient_profile.get('age', 0) >= 60
        elif risk_factor == "age_80_plus":
            return patient_profile.get('age', 0) >= 80
        elif risk_factor == "kidney_disease":
            conditions = (patient_profile.get('conditions') or '').lower()
            return 'kidney' in conditions or 'renal' in conditions
        elif risk_factor == "liver_disease":
            conditions = (patient_profile.get('conditions') or '').lower()
            return 'liver' in conditions or 'hepatic' in conditions
        elif risk_factor == "heart_disease":
            conditions = (patient_profile.get('conditions') or '').lower()
            return 'heart' in conditions or 'cardiac' in conditions
        elif risk_factor == "diabetes":
            conditions = (patient_profile.get('conditions') or '').lower()
            return 'diabetes' in conditions
        return False

side_effects_db = SideEffectsDatabase('data/side_effects_database_drugbank.json')

# --- Multi-Drug Conflict Checker ---
class MultiDrugConflictChecker:
    def __init__(self, conflicts_file):
        try:
            self.conflicts_data = load_json_file(conflicts_file)
            print("[INFO] Multi-drug conflict checker initialized successfully.")
        except Exception as e: 
            print(f"[ERROR] Could not initialize multi-drug conflict checker: {e}")
            self.conflicts_data = {"triangular_conflicts": [], "category_conflicts": {}}
    
    def check_triangular_conflicts(self, new_drug, existing_meds):
        """Check for 3-drug conflicts (triangular conflicts)"""
        conflicts = []
        
        if len(existing_meds) < 2:
            return conflicts
        
        # Get all drug names
        all_drugs = [med['drug_name'].lower().strip() for med in existing_meds] + [new_drug.lower().strip()]
        
        # Check each triangular conflict
        for conflict in self.conflicts_data.get('triangular_conflicts', []):
            conflict_drugs = [drug.lower().strip() for drug in conflict['drugs']]
            
            # Check if all 3 drugs in conflict are present
            if all(drug in all_drugs for drug in conflict_drugs):
                conflicts.append({
                    'type': 'triangular',
                    'drugs': conflict['drugs'],
                    'severity': conflict['severity'],
                    'description': conflict['description'],
                    'warning': conflict['warning']
                })
        
        return conflicts
    
    def check_category_conflicts(self, new_drug, existing_meds):
        """Check for category-based conflicts"""
        conflicts = []
        
        # Get drug categories
        categories = self.conflicts_data.get('drug_categories', {})
        
        # Find categories for new drug
        new_drug_categories = []
        for category, drugs in categories.items():
            if any(drug in new_drug.lower() for drug in drugs):
                new_drug_categories.append(category)
        
        # Check against existing medications
        for med in existing_meds:
            med_categories = []
            for category, drugs in categories.items():
                if any(drug in med['drug_name'].lower() for drug in drugs):
                    med_categories.append(category)
            
            # Check for category conflicts
            for new_cat in new_drug_categories:
                for existing_cat in med_categories:
                    conflict_key = f"{new_cat}_{existing_cat}"
                    reverse_key = f"{existing_cat}_{new_cat}"
                    
                    if conflict_key in self.conflicts_data.get('category_conflicts', {}):
                        conflict_info = self.conflicts_data['category_conflicts'][conflict_key]
                    elif reverse_key in self.conflicts_data.get('category_conflicts', {}):
                        conflict_info = self.conflicts_data['category_conflicts'][reverse_key]
                    else:
                        continue
                    
                    conflicts.append({
                        'type': 'category',
                        'drugs': [new_drug, med['drug_name']],
                        'severity': conflict_info['severity'],
                        'description': conflict_info['description'],
                        'warning': conflict_info['warning']
                    })
        
        return conflicts

multi_drug_checker = MultiDrugConflictChecker('data/multi_drug_conflicts.json')

# --- Data Models ---
@dataclass
class Intent:
    """Result from intent classification"""
    type: str  # "conversational" | "medical"
    confidence: float  # 0.0 to 1.0
    extracted_drugs: List[str]  # Drug names found in message

@dataclass
class InteractionResult:
    """Standardized result from drug interaction analysis"""
    gnn_risk: float  # 0-100
    rag_interactions: List[Dict]  # Documented interactions from RAG
    llm_explanation: str  # Human-readable explanation
    verdict: str  # "SAFE TO ADD" | "CAUTION ADVISED" | "DO NOT ADD"
    can_add: bool  # True if safe to add
    dosage_validation: Dict  # Dosage safety information
    timestamp: str  # ISO format timestamp

# --- Interaction Engine (Unified Analysis Core) ---
class InteractionEngine:
    """Unified drug interaction analysis using GNN + RAG + LLM pipeline"""
    
    def __init__(self, gnn_model, drug_map, rag_system, dosage_validator, side_effects_db, multi_drug_checker):
        """Initialize with required dependencies"""
        self.gnn_model = gnn_model
        self.drug_map = drug_map
        self.rag_system = rag_system
        self.dosage_validator = dosage_validator
        self.side_effects_db = side_effects_db
        self.multi_drug_checker = multi_drug_checker
        print("[INFO] InteractionEngine initialized successfully.")
    
    def analyze_interaction(
        self,
        new_drug: str,
        existing_drugs: List[str],
        patient_profile: Optional[Dict] = None,
        dosage_info: Optional[Dict] = None
    ) -> InteractionResult:
        """
        Analyze drug interactions using GNN + RAG + LLM with graceful degradation
        
        Args:
            new_drug: The drug being added
            existing_drugs: List of existing drug names
            patient_profile: Optional patient information (age, conditions, etc.)
            dosage_info: Optional dosage details (amount, unit, frequency)
            
        Returns:
            InteractionResult containing all analysis results
        """
        # Track which components succeeded
        gnn_available = False
        rag_available = False
        llm_available = False
        
        try:
            # Handle single drug case (no existing drugs)
            if not existing_drugs or len(existing_drugs) == 0:
                return self._handle_single_drug(new_drug, patient_profile, dosage_info)
            
            # Step 1: GNN Prediction (with error handling)
            gnn_risk = 0.0
            try:
                gnn_risk = self._get_gnn_prediction(new_drug, existing_drugs)
                if gnn_risk > 0:
                    gnn_available = True
                    print(f"[SUCCESS] GNN prediction: {gnn_risk:.1f}%")
                else:
                    print("[WARNING] GNN prediction returned 0, continuing with RAG + LLM")
            except Exception as gnn_error:
                print(f"[ERROR] GNN component failed: {gnn_error}")
                gnn_risk = 0.0
            
            # Step 2: RAG System Lookup (with error handling)
            rag_interactions = []
            try:
                rag_interactions = self._get_rag_interactions(new_drug, existing_drugs)
                if rag_interactions:
                    rag_available = True
                    print(f"[SUCCESS] RAG found {len(rag_interactions)} interactions")
                else:
                    print("[WARNING] RAG found no interactions, continuing with GNN + LLM")
            except Exception as rag_error:
                print(f"[ERROR] RAG component failed: {rag_error}")
                rag_interactions = []
            
            # Step 3: Build comprehensive context
            try:
                context = self._build_context(
                    new_drug, 
                    existing_drugs, 
                    gnn_risk, 
                    rag_interactions,
                    patient_profile,
                    dosage_info
                )
            except Exception as context_error:
                print(f"[ERROR] Context building failed: {context_error}")
                # Build minimal context
                context = f"New Drug: {new_drug}\nExisting Drugs: {', '.join(existing_drugs)}\nGNN Risk: {gnn_risk:.1f}%"
            
            # Step 4: Get LLM explanation (with error handling and timeout)
            llm_explanation = ""
            try:
                llm_explanation = self._get_llm_explanation(context)
                if llm_explanation and len(llm_explanation.strip()) > 0:
                    llm_available = True
                    print("[SUCCESS] LLM explanation generated")
                else:
                    print("[WARNING] LLM returned empty response")
            except Exception as llm_error:
                print(f"[ERROR] LLM component failed: {llm_error}")
                llm_explanation = generate_fallback_response(context)
            
            # Add component availability notice to explanation if any failed
            if not gnn_available or not rag_available or not llm_available:
                availability_notice = "\n\n⚠️ Note: "
                failed_components = []
                if not gnn_available:
                    failed_components.append("AI risk prediction")
                if not rag_available:
                    failed_components.append("database lookup")
                if not llm_available:
                    failed_components.append("detailed analysis")
                
                if failed_components:
                    availability_notice += f"{', '.join(failed_components)} unavailable. "
                    availability_notice += "Analysis based on available components. Please consult a healthcare professional."
                    llm_explanation += availability_notice
            
            # Step 5: Determine verdict
            try:
                verdict = self._determine_verdict(llm_explanation, gnn_risk, rag_interactions)
            except Exception as verdict_error:
                print(f"[ERROR] Verdict determination failed: {verdict_error}")
                # Conservative default
                verdict = "DO NOT ADD"
            
            # Step 6: Validate dosage if provided
            dosage_validation = {}
            try:
                dosage_validation = self._validate_dosage(new_drug, dosage_info)
            except Exception as dosage_error:
                print(f"[ERROR] Dosage validation failed: {dosage_error}")
                dosage_validation = {
                    'is_safe': False,
                    'warnings': ['Dosage validation failed'],
                    'max_daily': None,
                    'max_single': None
                }
            
            # Step 7: Determine if can add (both interaction and dosage must be safe)
            interaction_safe = "SAFE TO ADD" in verdict
            dosage_safe = dosage_validation.get('is_safe', True)
            can_add = interaction_safe and dosage_safe
            
            return InteractionResult(
                gnn_risk=round(gnn_risk, 1),
                rag_interactions=rag_interactions,
                llm_explanation=llm_explanation,
                verdict=verdict,
                can_add=can_add,
                dosage_validation=dosage_validation,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"[ERROR] InteractionEngine critical error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback with conservative verdict
            return InteractionResult(
                gnn_risk=0.0,
                rag_interactions=[],
                llm_explanation=f"Unable to complete analysis due to system error. Please consult a healthcare professional immediately. Error: {str(e)}",
                verdict="DO NOT ADD",
                can_add=False,
                dosage_validation={'is_safe': False, 'warnings': ['Analysis failed - system error']},
                timestamp=datetime.now().isoformat()
            )
    
    def _handle_single_drug(self, drug: str, patient_profile: Optional[Dict], dosage_info: Optional[Dict]) -> InteractionResult:
        """Handle case where there are no existing drugs"""
        dosage_validation = self._validate_dosage(drug, dosage_info)
        
        explanation = f"{drug} appears safe when taken alone. "
        if patient_profile:
            explanation += f"Always consult your healthcare provider, especially given your medical history. "
        else:
            explanation += "Always consult your healthcare provider before starting any medication. "
        
        if not dosage_validation.get('is_safe', True):
            explanation += "However, there are dosage concerns - please review the warnings."
        
        return InteractionResult(
            gnn_risk=0.0,
            rag_interactions=[],
            llm_explanation=explanation,
            verdict="SAFE TO ADD" if dosage_validation.get('is_safe', True) else "DO NOT ADD",
            can_add=dosage_validation.get('is_safe', True),
            dosage_validation=dosage_validation,
            timestamp=datetime.now().isoformat()
        )
    
    def _get_gnn_prediction(self, new_drug: str, existing_drugs: List[str]) -> float:
        """Get GNN risk prediction for drug interactions with error handling"""
        try:
            if not self.gnn_model or not self.drug_map:
                print("[WARNING] GNN model or drug_map not available")
                return 0.0
            
            # Convert drug names to indices
            new_drug_idx = None
            existing_drug_indices = []
            
            # Find new drug index
            new_drug_lower = new_drug.lower().strip()
            for drug_name, idx in self.drug_map.items():
                if new_drug_lower in drug_name.lower() or drug_name.lower() in new_drug_lower:
                    new_drug_idx = idx
                    break
            
            if new_drug_idx is None:
                print(f"[WARNING] Drug '{new_drug}' not found in drug_map")
                return 0.0
            
            # Find existing drug indices
            for existing_drug in existing_drugs:
                existing_drug_lower = existing_drug.lower().strip()
                for drug_name, idx in self.drug_map.items():
                    if existing_drug_lower in drug_name.lower() or drug_name.lower() in existing_drug_lower:
                        existing_drug_indices.append(idx)
                        break
            
            if not existing_drug_indices:
                print("[WARNING] No existing drugs found in drug_map")
                return 0.0
            
            # Calculate average risk across all existing medications
            total_risk = 0.0
            count = 0
            
            for existing_idx in existing_drug_indices:
                try:
                    # Create edge index for this pair
                    edge_index = torch.tensor([[new_drug_idx], [existing_idx]], dtype=torch.long)
                    
                    # Get node embeddings
                    with torch.no_grad():
                        all_nodes = torch.tensor(list(range(len(self.drug_map))), dtype=torch.long)
                        edge_index_full = torch.tensor([[new_drug_idx, existing_idx], [existing_idx, new_drug_idx]], dtype=torch.long)
                        
                        # Get predictions
                        z = self.gnn_model.encode(all_nodes, edge_index_full)
                        pred = self.gnn_model.decode(z, edge_index)
                        risk_score = torch.sigmoid(pred).item()
                        
                        total_risk += risk_score
                        count += 1
                except Exception as pair_error:
                    print(f"[ERROR] GNN prediction failed for drug pair: {pair_error}")
                    # Continue with other pairs
                    continue
            
            if count > 0:
                avg_risk = (total_risk / count) * 100
                return min(avg_risk, 100.0)  # Cap at 100%
            
            return 0.0
            
        except Exception as e:
            print(f"[ERROR] GNN prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # Return 0 to allow system to continue with RAG + LLM
            return 0.0
    
    def _get_rag_interactions(self, new_drug: str, existing_drugs: List[str]) -> List[Dict]:
        """Get documented interactions from RAG system with error handling"""
        try:
            if not self.rag_system or not self.rag_system.df is not None:
                print("[WARNING] RAG system not available")
                return []
            
            interactions = []
            
            for existing_drug in existing_drugs:
                try:
                    interaction = self.rag_system.search_interaction(new_drug, existing_drug)
                    if interaction:
                        interactions.append(interaction)
                except Exception as search_error:
                    print(f"[ERROR] RAG search failed for {new_drug} and {existing_drug}: {search_error}")
                    # Continue with other drugs
                    continue
            
            return interactions
            
        except Exception as e:
            print(f"[ERROR] RAG lookup failed: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list to allow system to continue with GNN + LLM
            return []
    
    def _build_context(
        self, 
        new_drug: str, 
        existing_drugs: List[str],
        gnn_risk: float,
        rag_interactions: List[Dict],
        patient_profile: Optional[Dict],
        dosage_info: Optional[Dict]
    ) -> str:
        """Build comprehensive context for LLM"""
        context = ""
        
        # Patient profile
        if patient_profile:
            patient_age = calculate_age(patient_profile.get('dob'))
            name = patient_profile.get('name', 'Patient')
            conditions = patient_profile.get('conditions', 'None reported')
            allergies = patient_profile.get('drug_allergies', 'None reported')
            context += f"Patient Profile: Name: {name}, Age: {patient_age}, Conditions: {conditions}, "
            context += f"Allergies: {allergies}.\n"
        else:
            context += "Patient Profile: Anonymous user (no profile available).\n"
        
        # Current medications
        context += f"Current Medications: {', '.join(existing_drugs)}.\n"
        context += f"New Drug to Analyze: {new_drug}.\n\n"
        
        # GNN prediction
        context += f"GNN Predicted Risk: {gnn_risk:.1f}% chance of interaction\n\n"
        
        # Side effects
        if patient_profile:
            side_effects_info = self.side_effects_db.get_side_effects(new_drug, patient_profile)
        else:
            side_effects_info = self.side_effects_db.get_side_effects(new_drug, None)
        
        if side_effects_info.get('side_effects'):
            context += f"Common Side Effects of {new_drug}:\n"
            for effect in side_effects_info['side_effects'][:3]:
                context += f"- {effect}\n"
            context += "\n"
        
        if side_effects_info.get('risk_warnings'):
            context += f"Personalized Risk Warnings:\n"
            for warning in side_effects_info['risk_warnings']:
                context += f"- {warning}\n"
            context += "\n"
        
        # Multi-drug conflicts
        if patient_profile:
            # Create mock medication list for multi-drug checker
            mock_meds = [{'drug_name': drug} for drug in existing_drugs]
            triangular_conflicts = self.multi_drug_checker.check_triangular_conflicts(new_drug, mock_meds)
            category_conflicts = self.multi_drug_checker.check_category_conflicts(new_drug, mock_meds)
            
            if triangular_conflicts:
                context += "Multi-Drug Conflict Analysis (Triangular Conflicts):\n"
                for conflict in triangular_conflicts:
                    context += f"- {conflict['warning']}\n"
                    context += f"  Drugs involved: {', '.join(conflict['drugs'])}\n"
                    context += f"  Severity: {conflict['severity']}\n"
                context += "\n"
            
            if category_conflicts:
                context += "Category-Based Conflict Analysis:\n"
                for conflict in category_conflicts:
                    context += f"- {conflict['warning']}\n"
                context += "\n"
        
        # Dosage analysis (kept neutral, no hardcoded suggested dose line)
        if dosage_info and patient_profile:
            dosage_amount = dosage_info.get('dosage_amount')
            if dosage_amount:
                context += "Dosage Analysis:\n"
                context += f"- Entered dose: {dosage_amount}\n"
                context += "- Use FDA label text for direct verification.\n\n"
        
        # RAG interactions
        context += "Known Pairwise Interactions from Database (for reference):\n"
        if rag_interactions:
            for interaction in rag_interactions:
                context += f"- {interaction['drug_a']} and {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}\n"
        else:
            context += "- No specific pairwise interactions were found in the knowledge base.\n"
        
        return context
    
    def _get_llm_explanation(self, context: str) -> str:
        """Get LLM explanation using existing ask_local_llm function with timeout and error handling"""
        try:
            # Set a timeout for LLM API calls (handled in ask_local_llm)
            explanation = ask_local_llm(context)
            
            # Debug: Print what we got from the AI
            print(f"[DEBUG] AI Response length: {len(explanation) if explanation else 0}")
            print(f"[DEBUG] AI Response preview: {explanation[:200] if explanation else 'None'}...")
            
            if not explanation or len(explanation.strip()) == 0:
                print("[WARNING] LLM returned empty response, using fallback")
                return generate_fallback_response(context)
            
            return explanation
            
        except requests.exceptions.Timeout as timeout_error:
            print(f"[ERROR] LLM request timed out: {timeout_error}")
            return generate_fallback_response(context)
        except requests.exceptions.RequestException as req_error:
            print(f"[ERROR] LLM request failed: {req_error}")
            return generate_fallback_response(context)
        except Exception as e:
            print(f"[ERROR] LLM explanation failed: {e}")
            import traceback
            traceback.print_exc()
            # Use fallback response generator
            return generate_fallback_response(context)
    
    def _determine_verdict(self, llm_explanation: str, gnn_risk: float, rag_interactions: List[Dict]) -> str:
        """
        Determine verdict using deterministic logic based on GNN risk and RAG interactions.
        This ensures consistency across multiple calls with the same inputs.
        LLM explanation is for human readability only and doesn't affect the verdict.
        
        Verdict rules:
        - Risk > 70: "DO NOT ADD"
        - Risk 30-70 (inclusive): "CAUTION ADVISED"
        - Risk < 30: "SAFE TO ADD"
        - RAG severity overrides GNN risk
        """
        # Use deterministic logic based on risk score and interactions
        # Check RAG interactions for severity first (highest priority)
        for interaction in rag_interactions:
            severity = interaction.get('severity', '').lower()
            if severity in ['major', 'severe', 'contraindicated']:
                return "DO NOT ADD"
            elif severity in ['moderate', 'moderate risk']:
                return "CAUTION ADVISED"
        
        # Then check GNN risk score
        if gnn_risk > 70:
            return "DO NOT ADD"
        elif gnn_risk >= 30:  # Changed from > 30 to >= 30
            return "CAUTION ADVISED"
        
        return "SAFE TO ADD"
    
    def _validate_dosage(self, drug: str, dosage_info: Optional[Dict]) -> Dict:
        """Validate dosage using dosage validator with error handling"""
        if not dosage_info:
            return {
                'is_safe': True,
                'warnings': [],
                'max_daily': None,
                'max_single': None
            }
        
        try:
            dosage_amount = dosage_info.get('dosage_amount')
            dosage_unit = dosage_info.get('dosage_unit', 'mg')
            frequency = dosage_info.get('frequency', 'daily')
            
            return self.dosage_validator.validate_dosage(drug, dosage_amount, dosage_unit, frequency)
        except Exception as e:
            print(f"[ERROR] Dosage validation failed: {e}")
            import traceback
            traceback.print_exc()
            # Return conservative result
            return {
                'is_safe': False,
                'warnings': ['Dosage validation failed'],
                'max_daily': None,
                'max_single': None
            }

# Initialize global InteractionEngine instance
try:
    interaction_engine = InteractionEngine(
        gnn_model=gnn_model,
        drug_map=drug_map,
        rag_system=rag_system,
        dosage_validator=dosage_validator,
        side_effects_db=side_effects_db,
        multi_drug_checker=multi_drug_checker
    )
    print("[SUCCESS] InteractionEngine global instance initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize InteractionEngine: {e}")
    interaction_engine = None

# --- Intent Classifier (Message Classification) ---
class IntentClassifier:
    """Classify user messages as conversational or medical"""
    
    def __init__(self, drug_map):
        """Initialize with drug map for drug name detection"""
        self.drug_map = drug_map
        print("[INFO] IntentClassifier initialized successfully.")
    
    def classify(self, message: str) -> Intent:
        """
        Classify user message intent
        
        Args:
            message: User's input text
            
        Returns:
            Intent object with type, confidence, and extracted_drugs
        """
        if not message or not message.strip():
            return Intent(type="conversational", confidence=0.5, extracted_drugs=[])
        
        message_lower = message.lower().strip()
        
        # Check conversational patterns first
        conversational_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?]*$',
            r'^(h+i+|he+y+|hello+|hlo+|yo+|sup+)[\s!?]*$',
            r'^(bye|goodbye|see you|thanks|thank you|thx)[\s!?]*$',
            r'^how are you',
            r'^what\'?s up',
            r'^who are you',
            r'^what can you do',
            r'^help[\s!?]*$',
            r'^how\'?s it going',
            r'^what\'?s going on',
        ]
        
        for pattern in conversational_patterns:
            if re.match(pattern, message_lower):
                return Intent(type="conversational", confidence=0.95, extracted_drugs=[])
        
        # Check for drug names
        extracted_drugs = self._extract_drug_names(message)
        
        # Check for medical keywords
        medical_keywords = [
            'take', 'medication', 'drug', 'interaction', 'safe', 
            'combine', 'dosage', 'side effect', 'prescription',
            'medicine', 'pill', 'tablet', 'capsule', 'dose',
            'symptoms', 'treatment', 'pharmacy', 'doctor',
            'health', 'medical', 'adverse', 'reaction'
        ]
        has_medical_keywords = any(keyword in message_lower for keyword in medical_keywords)
        
        # Determine intent
        if extracted_drugs:
            # Drug names found - definitely medical
            return Intent(type="medical", confidence=0.95, extracted_drugs=extracted_drugs)
        elif has_medical_keywords:
            # Medical keywords but no drug names - likely medical
            return Intent(type="medical", confidence=0.85, extracted_drugs=[])
        
        # Default to conversational for ambiguous cases
        return Intent(type="conversational", confidence=0.6, extracted_drugs=[])
    
    def _extract_drug_names(self, message: str) -> List[str]:
        """Extract drug names from message using drug_map"""
        if not self.drug_map:
            return []
        
        extracted = []
        message_lower = message.lower().strip()
        
        # Check each drug in drug_map
        for drug_name in self.drug_map.keys():
            drug_lower = drug_name.lower()
            
            # Check if drug name appears in message
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(drug_lower) + r'\b'
            if re.search(pattern, message_lower):
                extracted.append(drug_name)
        
        # Also check for common drug name aliases/variations
        # This helps match common names like "aspirin" to "Acetylsalicylic acid"
        common_aliases = {
            'aspirin': 'Acetylsalicylic acid',
            'tylenol': 'Acetaminophen',
            'paracetamol': 'Acetaminophen',
            'advil': 'Ibuprofen',
            'motrin': 'Ibuprofen',
        }
        
        for alias, drug_name in common_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, message_lower) and drug_name in self.drug_map:
                if drug_name not in extracted:
                    extracted.append(drug_name)
        
        return extracted

# Initialize global IntentClassifier instance
try:
    intent_classifier = IntentClassifier(drug_map=drug_map)
    print("[SUCCESS] IntentClassifier global instance initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize IntentClassifier: {e}")
    intent_classifier = None

# --- Conversational Handler (Casual Response Generator) ---
class ConversationalHandler:
    """Generate appropriate responses for conversational intents"""
    
    def __init__(self):
        """Initialize with response templates"""
        self.response_templates = {
            'greeting': [
                "Hi! 👋 I'm your AI Health Assistant. I can help you check drug interactions and answer medication questions. What would you like to know?",
                "Hello! 👋 I'm here to help with your medication questions. How can I assist you today?",
                "Hey there! 👋 I'm Dr. MediBot, your health assistant. Ask me about drug interactions or medication safety!",
                "Hi! 😊 I'm your friendly AI health assistant. I specialize in checking drug interactions. What can I help you with?",
                "Hello! 🤖 I'm here to help keep you safe with your medications. What would you like to know?"
            ],
            'farewell': [
                "Take care! 👋 Feel free to come back anytime you have medication questions.",
                "Goodbye! Stay healthy and don't hesitate to reach out if you need help. 👋",
                "See you later! Remember, I'm always here to help with your medication questions. 👋",
                "Bye! 👋 Stay safe and healthy. Come back anytime you need medication advice!",
                "Take care! 😊 Don't hesitate to ask if you have any medication concerns in the future."
            ],
            'how_are_you': [
                "I'm doing great, thanks for asking! 😊 I'm here and ready to help with your medication questions. What would you like to know?",
                "I'm functioning perfectly! 🤖 More importantly, how can I help you with your health questions today?",
                "I'm doing well, thank you! 😊 I'm always ready to help with medication safety. What can I do for you?",
                "I'm great! 🤖 Thanks for asking. Now, how can I help you stay safe with your medications?"
            ],
            'who_are_you': [
                "I'm Dr. MediBot, your AI health assistant! 🤖 I specialize in checking drug interactions and providing medication safety information. I use advanced AI models to analyze potential risks when combining medications. How can I help you today?",
                "I'm an AI-powered health assistant designed to help you understand drug interactions and medication safety. I analyze your medications using machine learning and medical databases to keep you safe. What would you like to know?",
                "I'm your AI health assistant! 😊 I help people check if their medications are safe to take together. I use advanced technology to analyze drug interactions. What can I help you with?",
                "I'm Dr. MediBot! 🤖 I'm here to help you stay safe with your medications by checking for dangerous drug interactions. Ask me anything about medication safety!"
            ],
            'help': [
                "I can help you with:\n\n✅ Checking drug interactions\n✅ Analyzing medication safety\n✅ Providing dosage information\n✅ Explaining side effects\n\nJust ask me something like: 'Can I take aspirin with ibuprofen?' or 'Is paracetamol safe?'",
                "Here's what I can do for you:\n\n💊 Check if two or more drugs are safe to take together\n📊 Analyze interaction risks using AI\n⚠️ Warn you about potential dangers\n📋 Provide detailed safety information\n\nTry asking: 'Can I combine [drug1] and [drug2]?'",
                "I'm here to help with medication safety! I can:\n\n🔍 Check drug interactions\n💊 Analyze medication combinations\n⚠️ Identify potential risks\n📊 Provide personalized safety advice\n\nJust ask me about any medications you're concerned about!",
                "Great question! Here's how I can help:\n\n✨ Check if medications are safe together\n✨ Analyze interaction risks\n✨ Provide dosage guidance\n✨ Explain side effects\n\nTry asking: 'Is it safe to take [drug name]?' or 'Can I combine [drug1] and [drug2]?'"
            ],
            'thank_you': [
                "You're welcome! 😊 Feel free to ask if you have any other medication questions.",
                "Happy to help! 👋 Don't hesitate to reach out if you need more assistance.",
                "My pleasure! 😊 I'm always here if you have more questions about your medications.",
                "You're very welcome! 🤖 Stay safe and healthy. Come back anytime!",
                "Glad I could help! 😊 Remember, I'm here whenever you need medication advice."
            ],
            'unclear': [
                "I'm not sure I understand. Are you asking about a specific medication or drug interaction? Feel free to ask me something like: 'Can I take aspirin with ibuprofen?'",
                "Could you clarify what you'd like to know? I'm here to help with medication questions and drug interactions. Try asking about specific drugs!",
                "I'm not quite sure what you're asking. I specialize in medication safety and drug interactions. Try asking: 'Is [drug name] safe?' or 'Can I combine [drug1] and [drug2]?'",
                "Hmm, I'm not sure I caught that. I'm best at answering questions about medications and drug interactions. What would you like to know about your medications?",
                "I didn't quite understand that. I'm here to help with medication safety! Try asking about specific drugs or drug combinations."
            ]
        }
        print("[INFO] ConversationalHandler initialized successfully.")
    
    def handle(self, message: str, intent: Intent) -> str:
        """
        Generate conversational response
        
        Args:
            message: User's input text
            intent: Classified intent
            
        Returns:
            Appropriate conversational response
        """
        message_lower = message.lower().strip()
        
        # Determine response category
        if self._is_greeting(message_lower):
            return self._get_random_response('greeting')
        elif self._is_farewell(message_lower):
            return self._get_random_response('farewell')
        elif self._is_how_are_you(message_lower):
            return self._get_random_response('how_are_you')
        elif self._is_who_are_you(message_lower):
            return self._get_random_response('who_are_you')
        elif self._is_help(message_lower):
            return self._get_random_response('help')
        elif self._is_thank_you(message_lower):
            return self._get_random_response('thank_you')
        else:
            return self._get_random_response('unclear')
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?]*$',
            r'^(h+i+|he+y+|hello+|hlo+|yo+|sup+)[\s!?]*$'
        ]
        return any(re.match(pattern, message) for pattern in greeting_patterns)
    
    def _is_farewell(self, message: str) -> bool:
        """Check if message is a farewell"""
        farewell_patterns = [
            r'^(bye|goodbye|see you|thanks|thank you|thx)[\s!?]*$',
            r'^(bye|goodbye)[\s!?]*$'
        ]
        return any(re.match(pattern, message) for pattern in farewell_patterns)
    
    def _is_how_are_you(self, message: str) -> bool:
        """Check if message is asking how are you"""
        return re.match(r'^how are you', message) or re.match(r'^how\'?s it going', message)
    
    def _is_who_are_you(self, message: str) -> bool:
        """Check if message is asking who are you"""
        return re.match(r'^who are you', message) or re.match(r'^what are you', message)
    
    def _is_help(self, message: str) -> bool:
        """Check if message is asking for help"""
        help_patterns = [
            r'^help[\s!?]*$',
            r'^what can you do',
            r'^what do you do',
            r'^how can you help'
        ]
        return any(re.match(pattern, message) for pattern in help_patterns)
    
    def _is_thank_you(self, message: str) -> bool:
        """Check if message is a thank you"""
        thank_patterns = [
            r'^(thanks|thank you|thx)[\s!?]*$',
            r'^thank you',
            r'^thanks'
        ]
        return any(re.match(pattern, message) for pattern in thank_patterns)
    
    def _get_random_response(self, category: str) -> str:
        """Get a random response from the specified category"""
        responses = self.response_templates.get(category, self.response_templates['unclear'])
        return random.choice(responses)

# Initialize global ConversationalHandler instance
try:
    conversational_handler = ConversationalHandler()
    print("[SUCCESS] ConversationalHandler global instance initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize ConversationalHandler: {e}")
    conversational_handler = None

# --- Response Format Validation ---
def validate_interaction_result(result: InteractionResult) -> bool:
    """
    Validate that InteractionResult has all required fields with correct types
    
    Args:
        result: InteractionResult to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check all required fields exist
        required_fields = ['gnn_risk', 'rag_interactions', 'llm_explanation', 
                          'verdict', 'can_add', 'dosage_validation', 'timestamp']
        
        for field in required_fields:
            if not hasattr(result, field):
                print(f"[VALIDATION ERROR] Missing field: {field}")
                return False
        
        # Validate field types
        if not isinstance(result.gnn_risk, (int, float)):
            print(f"[VALIDATION ERROR] gnn_risk must be numeric, got {type(result.gnn_risk)}")
            return False
        
        if not isinstance(result.rag_interactions, list):
            print(f"[VALIDATION ERROR] rag_interactions must be list, got {type(result.rag_interactions)}")
            return False
        
        if not isinstance(result.llm_explanation, str):
            print(f"[VALIDATION ERROR] llm_explanation must be string, got {type(result.llm_explanation)}")
            return False
        
        if not isinstance(result.verdict, str):
            print(f"[VALIDATION ERROR] verdict must be string, got {type(result.verdict)}")
            return False
        
        if not isinstance(result.can_add, bool):
            print(f"[VALIDATION ERROR] can_add must be bool, got {type(result.can_add)}")
            return False
        
        if not isinstance(result.dosage_validation, dict):
            print(f"[VALIDATION ERROR] dosage_validation must be dict, got {type(result.dosage_validation)}")
            return False
        
        if not isinstance(result.timestamp, str):
            print(f"[VALIDATION ERROR] timestamp must be string, got {type(result.timestamp)}")
            return False
        
        # Validate verdict values
        valid_verdicts = ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
        if result.verdict not in valid_verdicts:
            print(f"[VALIDATION ERROR] Invalid verdict: {result.verdict}")
            return False
        
        # Validate gnn_risk range
        if not (0 <= result.gnn_risk <= 100):
            print(f"[VALIDATION ERROR] gnn_risk must be 0-100, got {result.gnn_risk}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[VALIDATION ERROR] Exception during validation: {e}")
        return False

def validate_response_format(response_dict: Dict, endpoint_type: str) -> bool:
    """
    Validate that response dictionary has standardized format for the endpoint
    
    Args:
        response_dict: Response dictionary to validate
        endpoint_type: Type of endpoint ('quick_check', 'emergency_check', 'chatbot_medical', 'chatbot_conversational')
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if endpoint_type == 'quick_check':
            # Quick Check must have: gnn_risk, verdict, ai_response, can_add
            required = ['gnn_risk', 'verdict', 'ai_response', 'can_add']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Quick Check missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            if not isinstance(response_dict['verdict'], str):
                return False
            if not isinstance(response_dict['ai_response'], str):
                return False
            if not isinstance(response_dict['can_add'], bool):
                return False
            
        elif endpoint_type == 'emergency_check':
            # Emergency Check must have: status, response, gnn_risk, drug1, drug2
            required = ['status', 'response', 'gnn_risk', 'drug1', 'drug2']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Emergency Check missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['status'], str):
                return False
            if not isinstance(response_dict['response'], str):
                return False
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            
            # Validate status values
            valid_statuses = ['SAFE', 'CAUTION', 'UNSAFE']
            if response_dict['status'] not in valid_statuses:
                print(f"[VALIDATION ERROR] Invalid status: {response_dict['status']}")
                return False
            
        elif endpoint_type == 'chatbot_medical':
            # Chatbot medical must have: response, verdict, gnn_risk, intent
            required = ['response', 'verdict', 'gnn_risk', 'intent']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Chatbot medical missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['response'], str):
                return False
            if not isinstance(response_dict['verdict'], str):
                return False
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            if response_dict['intent'] != 'medical':
                print(f"[VALIDATION ERROR] Intent must be 'medical', got {response_dict['intent']}")
                return False
            
        elif endpoint_type == 'chatbot_conversational':
            # Chatbot conversational must have: response, intent
            required = ['response', 'intent']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Chatbot conversational missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['response'], str):
                return False
            if response_dict['intent'] != 'conversational':
                print(f"[VALIDATION ERROR] Intent must be 'conversational', got {response_dict['intent']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[VALIDATION ERROR] Exception during response validation: {e}")
        return False

# --- OpenFDA Dosage Text Summarization ---
def summarize_openfda_dosage(openfda_dosage_text, max_length=300):
    """
    Summarize long OpenFDA dosage text to avoid overwhelming the AI context.
    
    Args:
        openfda_dosage_text: Full OpenFDA dosage_and_administration text
        max_length: Maximum length of summary (default 300 characters)
    
    Returns:
        Summarized dosage text
    """
    try:
        if not openfda_dosage_text or openfda_dosage_text == "Dosage information not available.":
            return openfda_dosage_text
        
        # If text is already short, return as-is
        if len(openfda_dosage_text) <= max_length:
            return openfda_dosage_text
        
        import re
        
        # Try to extract key dosage information (first few sentences with numbers)
        # Look for sentences containing dosage numbers
        sentences = re.split(r'[.!?]\s+', openfda_dosage_text)
        
        key_sentences = []
        total_length = 0
        
        for sentence in sentences:
            # Prioritize sentences with dosage numbers (mg, mcg, etc.)
            if re.search(r'\d+\s*(?:mg|mcg|g|mL|units?)', sentence, re.IGNORECASE):
                if total_length + len(sentence) <= max_length:
                    key_sentences.append(sentence)
                    total_length += len(sentence) + 2  # +2 for ". "
                else:
                    break
        
        if key_sentences:
            summary = '. '.join(key_sentences) + '.'
            return summary
        
        # Fallback: just truncate to max_length
        return openfda_dosage_text[:max_length] + "..."
    
    except Exception as e:
        print(f"[ERROR] Error summarizing OpenFDA text: {e}")
        # Return truncated version as fallback
        return openfda_dosage_text[:max_length] + "..." if len(openfda_dosage_text) > max_length else openfda_dosage_text

# --- OpenFDA Dosage Comparison ---
FDA_DOSAGE_MISSING_TEXT = "Dosage information not available from FDA label"

def _extract_user_dose(user_input):
    """Extract numeric amount and unit from user input dict."""
    amount = user_input.get('dosage_amount')
    unit = (user_input.get('dosage_unit') or '').strip().lower()
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = None
    return amount, unit

def _extract_fda_units(openfda_text):
    supported_units = ["mg", "ml", "drops", "sprays", "tablet", "capsule"]
    text = (openfda_text or "").lower()
    found = set()
    for unit in supported_units:
        if re.search(rf"\b{re.escape(unit)}s?\b", text):
            found.add(unit)
    return found

def _extract_fda_values_for_unit(openfda_text, unit):
    text = openfda_text or ""
    values = []
    unit_pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}s?\b"
    for match in re.findall(unit_pattern, text, re.IGNORECASE):
        try:
            values.append(float(match))
        except ValueError:
            continue

    range_pattern = rf"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*{re.escape(unit)}s?\b"
    for low, high in re.findall(range_pattern, text, re.IGNORECASE):
        try:
            values.extend([float(low), float(high)])
        except ValueError:
            continue
    return values

def compare_dosage(user_input, openfda_text):
    """
    Compare user-entered dosage against OpenFDA dosage text with unit-aware logic.
    """
    try:
        if not openfda_text or openfda_text == "Dosage information not available.":
            return FDA_DOSAGE_MISSING_TEXT

        user_amount, user_unit = _extract_user_dose(user_input or {})
        fda_units = _extract_fda_units(openfda_text)

        if user_unit and fda_units and user_unit not in fda_units:
            return (
                "The entered dosage uses different units than FDA guidance "
                "(e.g., mg vs sprays). Direct comparison is not possible. "
                "Please refer to the FDA dosage information below."
            )

        if not user_unit or user_amount is None:
            return "FDA dosage guidance is provided below. Please verify manually."

        fda_values = _extract_fda_values_for_unit(openfda_text, user_unit)
        if not fda_values:
            return "FDA dosage guidance is provided below. Please verify manually."

        min_val = min(fda_values)
        max_val = max(fda_values)
        if min_val <= user_amount <= max_val:
            return "The entered dosage appears consistent with FDA guidance."
        if user_amount > max_val:
            return "The entered dosage may exceed recommended dosing."
        return "The entered dosage is lower than typical recommended dosing."
    except Exception as e:
        print(f"[ERROR] compare_dosage failed: {e}")
        return "FDA dosage guidance is provided below. Please verify manually."

# --- Personalized Dosage Adjustment ---
def calculate_personalized_dosage(drug_name, dosage_amount, patient_profile):
    """Calculate personalized dosage based on patient profile"""
    try:
        dosage_amount = float(dosage_amount)
    except (ValueError, TypeError):
        return {"suggested_dose": dosage_amount, "adjustment_reason": "Invalid dosage amount"}
    
    # Get base dosage limits
    dosage_validation = dosage_validator.validate_dosage(drug_name, dosage_amount, "mg", "daily")
    max_daily = dosage_validation.get('max_daily', dosage_amount)
    
    # Calculate adjustment factors
    adjustment_factor = 1.0
    adjustment_reasons = []
    
    # Age-based adjustments
    age = patient_profile.get('age', 0)
    if age >= 80:
        adjustment_factor *= 0.7
        adjustment_reasons.append("Age 80+ (reduced metabolism)")
    elif age >= 65:
        adjustment_factor *= 0.8
        adjustment_reasons.append("Age 65+ (reduced kidney function)")
    elif age >= 60:
        adjustment_factor *= 0.9
        adjustment_reasons.append("Age 60+ (mild kidney function decline)")
    
    # Condition-based adjustments
    conditions = (patient_profile.get('conditions') or '').lower()
    
    if 'kidney' in conditions or 'renal' in conditions:
        adjustment_factor *= 0.75
        adjustment_reasons.append("Kidney disease (reduced clearance)")
    
    if 'liver' in conditions or 'hepatic' in conditions:
        adjustment_factor *= 0.8
        adjustment_reasons.append("Liver disease (reduced metabolism)")
    
    if 'heart' in conditions or 'cardiac' in conditions:
        adjustment_factor *= 0.85
        adjustment_reasons.append("Heart disease (reduced tolerance)")
    
    # Weight-based adjustments
    weight = patient_profile.get('weight_kg')
    if weight:
        try:
            weight = float(weight)
            if weight < 50:  # Underweight
                adjustment_factor *= 0.8
                adjustment_reasons.append("Low body weight (reduced tolerance)")
            elif weight > 100:  # Overweight
                adjustment_factor *= 1.1
                adjustment_reasons.append("Higher body weight (increased clearance)")
        except (ValueError, TypeError):
            pass
    
    # Calculate suggested dose with safe handling of None values
    if max_daily is not None and max_daily > 0:
        suggested_dose = min(dosage_amount * adjustment_factor, max_daily)
        
        # Ensure minimum effective dose
        min_effective_dose = max_daily * 0.5  # At least 50% of max dose
        if suggested_dose < min_effective_dose:
            suggested_dose = min_effective_dose
            adjustment_reasons.append("Minimum effective dose maintained")
    else:
        # No max dose data available, just apply adjustment factor
        suggested_dose = dosage_amount * adjustment_factor
        if not adjustment_reasons:
            adjustment_reasons.append("No maximum dose data available for this medication")
    
    return {
        "suggested_dose": round(suggested_dose, 1),
        "original_dose": dosage_amount,
        "adjustment_factor": round(adjustment_factor, 2),
        "adjustment_reasons": adjustment_reasons,
        "max_safe_dose": max_daily if max_daily else "Unknown"
    }

def build_rule_based_safety_layer(
    new_drug,
    patient,
    existing_meds,
    gnn_risk,
    dosage_validation,
    dosage_amount=None,
    dosage_unit=None,
    frequency=None
):
    """
    Deterministic safety explanation layer.
    Guarantees side effects, multi-drug warnings, and personalized dosage guidance
    appear in output even if LLM response is weak.
    """
    lines = []
    hard_block = False
    caution = False

    def _safe_get(obj, key, default=None):
        """Support dict and sqlite3.Row access patterns."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return obj[key]
        except Exception:
            return default

    existing_meds = existing_meds or []
    med_names = []
    for med in existing_meds:
        drug_name = _safe_get(med, 'drug_name', '')
        if drug_name:
            med_names.append(drug_name)
    patient_age = calculate_age(patient.get('dob')) if patient else 0
    patient_conditions = (patient.get('conditions') or '').lower() if patient else ""

    # Pairwise interactions (deterministic DB lookup)
    pairwise_found = []
    for med_name in med_names:
        interaction = rag_system.search_interaction(new_drug, med_name)
        if interaction:
            pairwise_found.append(interaction)

    lines.append("Clinical Safety Layer:")
    if pairwise_found:
        lines.append(f"- Pairwise interactions found: {len(pairwise_found)}")
        for interaction in pairwise_found[:3]:
            severity = interaction.get('severity', 'unknown')
            lines.append(
                f"  - {interaction.get('drug_a')} + {interaction.get('drug_b')} ({severity}): "
                f"{interaction.get('interaction', 'Interaction documented in database.')}"
            )
            if str(severity).lower() in ["major", "severe", "contraindicated"]:
                hard_block = True
            else:
                caution = True
    else:
        lines.append("- Pairwise interactions: none documented in the current database for this combination.")

    # Multi-drug conflicts (triangular + category)
    mock_meds = [{'drug_name': n} for n in med_names]
    triangular_conflicts = multi_drug_checker.check_triangular_conflicts(new_drug, mock_meds)
    category_conflicts = multi_drug_checker.check_category_conflicts(new_drug, mock_meds)

    if triangular_conflicts:
        hard_block = True
        lines.append("- Multi-drug conflict detected (triangular):")
        for conflict in triangular_conflicts:
            lines.append(
                f"  - {', '.join(conflict.get('drugs', []))}: "
                f"{conflict.get('warning', conflict.get('description', 'Triangular conflict detected.'))}"
            )
    elif category_conflicts:
        caution = True
        lines.append("- Multi-drug category conflict detected:")
        for conflict in category_conflicts[:3]:
            lines.append(f"  - {conflict.get('warning', conflict.get('description', 'Category conflict detected.'))}")
    else:
        lines.append("- Multi-drug conflict: no triangular or category conflict detected.")

    # Side effects + risk explanation
    side_effects_info = side_effects_db.get_side_effects(new_drug, patient or {})
    common_side_effects = side_effects_info.get('side_effects') or []
    serious_side_effects = side_effects_info.get('serious_effects') or []
    risk_warnings = side_effects_info.get('risk_warnings') or []

    if common_side_effects:
        top_common = ", ".join(common_side_effects[:3])
        lines.append(f"- Side effect prediction: common effects include {top_common}.")
    else:
        lines.append("- Side effect prediction: no specific side-effect profile found in local database.")

    if dosage_validation and not dosage_validation.get('is_safe', True) and common_side_effects:
        lines.append(f"- High dose warning: current dose may increase risk of {common_side_effects[0]}.")
        caution = True

    if serious_side_effects:
        lines.append(f"- Serious effects to watch: {', '.join(serious_side_effects[:2])}.")

    if risk_warnings:
        lines.append("- Personalized risk flags:")
        for warning in risk_warnings[:3]:
            lines.append(f"  - {warning}")
        caution = True

    if dosage_validation and not dosage_validation.get('is_safe', True):
        caution = True

    # Keep model output as supportive signal only; never override evidence by itself
    lines.append(f"- GNN model risk analysis: predicted interaction score {gnn_risk:.1f}%.")

    # Additional patient-specific caution
    if any(k in patient_conditions for k in ["kidney", "renal", "liver", "hepatic", "heart", "cardiac"]):
        lines.append("- Existing chronic conditions increase sensitivity to interactions and dose issues.")
        caution = True

    return {
        "text": "\n".join(lines),
        "hard_block": hard_block,
        "caution": caution
    }

# --- Helper, Validation & AI Functions ---
def get_db_connection():
    conn = sqlite3.connect('medicine_log.db'); conn.row_factory = sqlite3.Row; return conn
def calculate_age(dob_str):
    if not dob_str: return 0
    try:
        birth_date = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError): return 0
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)
def is_strong_password(password):
    if len(password) < 8: return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password): return False, "Password must contain an uppercase letter."
    if not re.search(r"[a-z]", password): return False, "Password must contain a lowercase letter."
    if not re.search(r"[0-9]", password): return False, "Password must contain a number."
    if not re.search(r"[!@#$%^&*(),.?:{}|<>]", password): return False, "Password must contain a special character."
    return True, ""
def generate_fallback_response(context):
    """Generate a smart fallback response when LLM API is unavailable"""
    
    # Extract key information from context
    has_interactions = "No specific pairwise interactions were found" not in context
    gnn_risk_match = re.search(r'GNN Predicted Risk: ([\d.]+)%', context)
    gnn_risk = float(gnn_risk_match.group(1)) if gnn_risk_match else 0
    
    patient_name_match = re.search(r'Name: ([^,]+)', context)
    patient_name = patient_name_match.group(1) if patient_name_match else "there"
    
    drug_match = re.search(r'New Drug to Analyze: ([^\n]+)', context)
    new_drug = drug_match.group(1).strip() if drug_match else "this medication"
    
    # Check for dosage warnings
    has_dosage_warnings = "Dosage Warnings:" in context
    
    # Generate response based on analysis
    response = f"Hi {patient_name}! 👋\n\n"
    response += "I've completed a safety analysis for you using my medical database.\n\n"
    
    response += "AI Risk Analysis:\n"
    response += f"My AI system predicts a {gnn_risk:.1f}% interaction risk"
    if gnn_risk < 20:
        response += " - that's quite low and reassuring!\n\n"
    elif gnn_risk < 50:
        response += " - moderate risk, worth being cautious.\n\n"
    else:
        response += " - this is concerning and requires attention.\n\n"
    
    response += "Drug Interaction Check:\n"
    if has_interactions:
        response += f"I found documented interactions for {new_drug} in my database. Please review the detailed interaction information above carefully.\n\n"
    else:
        response += f"Good news! I didn't find any documented interactions between {new_drug} and your current medications in my database.\n\n"
    
    response += "Dosage Safety:\n"
    if has_dosage_warnings:
        response += "⚠️ I have concerns about the dosage you entered. Please review the dosage warnings above.\n\n"
    else:
        response += "The dosage appears to be within normal ranges based on my database.\n\n"
    
    response += "My Recommendation:\n"
    
    # Decision logic
    if gnn_risk > 70 or has_interactions or has_dosage_warnings:
        response += f"Based on my analysis, I recommend NOT adding {new_drug} without consulting your doctor first. "
        if has_interactions:
            response += "There are documented interaction risks. "
        if has_dosage_warnings:
            response += "The dosage may not be safe. "
        response += "Please discuss this with your healthcare provider.\n\n"
        verdict = "DO NOT ADD"
    elif gnn_risk > 40:
        response += f"The risk level suggests caution. I recommend discussing {new_drug} with your doctor before adding it to your regimen.\n\n"
        verdict = "DO NOT ADD"
    else:
        response += f"Based on my analysis, {new_drug} appears to be relatively safe to add. However, always monitor for any unusual symptoms and consult your doctor if you have concerns.\n\n"
        verdict = "SAFE TO ADD"
    
    response += f"Verdict: {verdict}"
    
    return response

# --- OpenFDA API Integration ---

# Global cache for OpenFDA dosage data
openfda_dosage_cache = {}
OPENROUTER_COOLDOWN_UNTIL = 0
OPENROUTER_COOLDOWN_REASON = ""

def get_cached_dosage(drug_name):
    """
    Get dosage from cache if available.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Cached dosage text if found, None otherwise
    """
    # Normalize drug name (lowercase, strip)
    normalized_name = drug_name.lower().strip()
    cached_value = openfda_dosage_cache.get(normalized_name)
    
    # Log cache hit
    if cached_value:
        print(f"[INFO] OpenFDA: Using cached dosage for {drug_name}")
    
    return cached_value

def cache_dosage(drug_name, dosage):
    """
    Store dosage in cache.
    
    Args:
        drug_name: Name of the drug
        dosage: Dosage text to cache
    """
    # Normalize drug name (lowercase, strip)
    normalized_name = drug_name.lower().strip()
    openfda_dosage_cache[normalized_name] = dosage

def fetch_openfda_dosage(drug_name):
    """
    Fetch dosage information from OpenFDA API.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Dosage text if found, None otherwise
        
    Behavior:
        - Queries OpenFDA drug label API
        - Extracts dosage_and_administration field
        - Handles timeouts, HTTP errors, and missing data
        - Returns cleaned text or None
    """
    try:
        # Log the attempt
        print(f"[INFO] OpenFDA: Fetching dosage for {drug_name}")
        
        base_url = "https://api.fda.gov/drug/label.json"
        drug_name_clean = (drug_name or "").strip()
        escaped_name = drug_name_clean.replace('"', '\\"')

        search_queries = [
            f'openfda.generic_name:"{escaped_name}"',
            f'openfda.brand_name:"{escaped_name}"',
            f'openfda.substance_name:"{escaped_name}"',
            f'openfda.generic_name:{drug_name_clean}',
            f'openfda.brand_name:{drug_name_clean}',
        ]

        for query in search_queries:
            response = requests.get(
                base_url,
                params={"search": query, "limit": 1},
                timeout=5
            )

            if response.status_code in (404, 429):
                continue
            if response.status_code >= 400:
                continue

            try:
                data = response.json()
            except json.JSONDecodeError:
                continue

            results = data.get('results') or []
            if not results:
                continue

            result = results[0]
            dosage_list = result.get('dosage_and_administration') or []
            if not dosage_list:
                continue

            # Keep FDA text unmodified; join multi-part sections verbatim.
            dosage_text = "\n\n".join([part for part in dosage_list if isinstance(part, str) and part.strip()])
            if dosage_text.strip():
                print(f"[SUCCESS] OpenFDA: Found dosage for {drug_name} using query: {query}")
                return dosage_text

        print(f"[WARNING] OpenFDA: No dosage text found for {drug_name}")
        return None
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] OpenFDA: Timeout fetching {drug_name}")
        return None
    except requests.exceptions.ConnectionError as conn_error:
        print(f"[ERROR] OpenFDA: Connection error for {drug_name}: {conn_error}")
        return None
    except requests.exceptions.RequestException as req_error:
        print(f"[ERROR] OpenFDA: Request error for {drug_name}: {req_error}")
        return None
    except Exception as e:
        print(f"[ERROR] OpenFDA: Unexpected error for {drug_name}: {e}")
        return None

def check_drugbank_dosage(drug_name):
    """
    Check if DrugBank has dosage information for the drug.
    
    Args:
        drug_name: Name of the drug
        
    Returns:
        Dosage text if found in DrugBank, None otherwise
    """
    try:
        # Normalize drug name for matching
        drug_name_lower = drug_name.lower().strip()
        
        # Return None for empty drug name
        if not drug_name_lower:
            print(f"[INFO] DrugBank: No dosage found for {drug_name}")
            return None
        
        # Search through dosage_limits dictionary for case-insensitive partial match
        for drug_key, drug_info in dosage_validator.dosage_limits.items():
            # Perform case-insensitive partial match
            if drug_name_lower in drug_key.lower() or drug_key.lower() in drug_name_lower:
                # Extract and format dosage information
                max_daily = drug_info.get("max_daily_mg") or drug_info.get("max_daily_iu") or drug_info.get("max_daily_units")
                max_single = drug_info.get("max_single_mg") or drug_info.get("max_single_iu") or drug_info.get("max_single_units")
                unit = drug_info.get("unit", "units")
                
                # Format dosage text
                dosage_text = f"Maximum daily dose: {max_daily} {unit}. Maximum single dose: {max_single} {unit}."
                
                # Add warnings if available
                warnings = drug_info.get("warnings", [])
                if warnings:
                    dosage_text += f" Warnings: {' '.join(warnings)}"
                
                print(f"[INFO] DrugBank: Found dosage for {drug_name}")
                return dosage_text
        
        # No match found
        print(f"[INFO] DrugBank: No dosage found for {drug_name}")
        return None
        
    except Exception as e:
        print(f"[ERROR] DrugBank dosage check failed for {drug_name}: {e}")
        return None

def get_drug_dosage_info(drug_name):
    """
    Get dosage information with fallback logic.
    
    Args:
        drug_name: Name of the drug
        
    Returns:
        Dictionary with:
            'dosage': str - Dosage text or "Dosage information not available."
            'dosage_source': str - "DrugBank" | "OpenFDA" | "None"
    """
    # Step 1: Check DrugBank
    drugbank_dosage = check_drugbank_dosage(drug_name)
    if drugbank_dosage:
        return {
            'dosage': drugbank_dosage,
            'dosage_source': 'DrugBank'
        }
    
    # Step 2: Check cache
    cached_dosage = get_cached_dosage(drug_name)
    if cached_dosage:
        print(f"[INFO] Using cached OpenFDA dosage for {drug_name}")
        return {
            'dosage': cached_dosage,
            'dosage_source': 'OpenFDA'
        }
    
    # Step 3: Call OpenFDA API
    print(f"[INFO] Fetching dosage from OpenFDA for {drug_name}")
    openfda_dosage = fetch_openfda_dosage(drug_name)
    
    if openfda_dosage:
        cache_dosage(drug_name, openfda_dosage)
        return {
            'dosage': openfda_dosage,
            'dosage_source': 'OpenFDA'
        }
    
    # Step 4: No data available
    return {
        'dosage': FDA_DOSAGE_MISSING_TEXT,
        'dosage_source': 'None'
    }

def ask_local_llm(context):
    # OpenRouter cooldown gate to avoid repeating the same billing/config errors every request
    global OPENROUTER_COOLDOWN_UNTIL, OPENROUTER_COOLDOWN_REASON

    prompt = f"""You are a personal AI health assistant - a knowledgeable, caring, and direct friend who provides COMPLETE personalized medication safety analysis.

[CRITICAL FORMATTING RULES]
- Write in PLAIN TEXT only (NO markdown, NO **, NO ##, NO strikethrough)
- Use simple dashes (-) for lists if needed
- Write as if talking to a friend in everyday language

[YOUR COMPLETE ANALYSIS DATA]
{context}

[YOUR MISSION: PROVIDE COMPLETE PERSONALIZED ANALYSIS]
You MUST provide a COMPREHENSIVE analysis covering ALL of these aspects:
1. GNN AI Risk Score (if available)
2. Pairwise Drug Interactions from database (if found)
3. Dosage Safety Assessment (if provided)
4. Side Effects relevant to this patient (if available)
5. Patient-specific considerations (age, conditions, medications)

DO NOT skip any section that has data available. Weave everything together naturally.

[MANDATORY STRUCTURE - INCLUDE ALL SECTIONS WITH DATA]

SECTION 1: WARM PERSONAL GREETING
- Start with "Hi [Patient Name]!"
- Brief friendly opening that acknowledges their specific situation

SECTION 2: AI RISK PREDICTION (ALWAYS INCLUDE IF AVAILABLE)
- Look for "GNN Predicted Risk" or "AI Risk Score" in the data
- State it clearly and naturally
- Explain what the percentage means (low/moderate/high)
- Examples:
  * "My AI analysis shows a 92% interaction risk - that's very high and concerning."
  * "The AI risk prediction is 8% - that's quite low and reassuring."
  * "My system calculated a 45% risk level - that's moderate, so we need to be careful."

SECTION 3: PAIRWISE DRUG INTERACTIONS (ALWAYS INCLUDE IF FOUND)
- Look for "Known Pairwise Interactions from Database" or "Factual Interactions"
- For EACH interaction, you MUST explain:
  a) Which two drugs interact
  b) What CONSEQUENCE happens (the actual health risk)
  c) WHY it matters for THIS patient
- Examples of explaining consequences:
  * "Warfarin and Ibuprofen interact - this combination significantly increases your risk of internal bleeding because both affect blood clotting"
  * "Metformin and Alcohol can cause dangerous drops in blood sugar, leading to dizziness, confusion, or even loss of consciousness"
  * "Lisinopril and Potassium supplements can cause your potassium levels to become dangerously high, affecting your heart rhythm"
- If NO interactions found, state clearly:
  * "I searched my database for interactions between [Drug] and your current medications ([list them]), and I found no documented interactions. This is good news."

SECTION 4: DOSAGE SAFETY (ALWAYS INCLUDE IF PROVIDED)
- Look for "Dosage Information" in the data
- Compare the entered dose against safe limits
- State clearly if it's safe or dangerous
- Examples:
  * "Your dosage of 100mg once daily is well within safe limits. The maximum safe daily dose is 4000mg, so you're using a very conservative amount."
  * "WARNING: Your dose of 600mg daily exceeds the safe limit of 400mg. This puts you at risk of liver damage."
  * "The 10mg you're planning to take is perfect - it's the standard safe dose for someone your age."

SECTION 5: SIDE EFFECTS (ALWAYS INCLUDE IF AVAILABLE)
- Look for "Side Effects" or "Common Side Effects" in the data
- List 2-4 relevant side effects in simple terms
- PERSONALIZE based on patient's age, conditions, or other medications
- Examples:
  * "For you at age 68, this medication may cause dizziness or drowsiness. Be extra careful when standing up or driving."
  * "Given your kidney condition, watch for swelling in your legs or feet - that could be a sign this medication isn't right for you."
  * "Common side effects include mild stomach upset and headache. Since you have a sensitive stomach, take this with food."

SECTION 6: PATIENT-SPECIFIC SYNTHESIS
- Bring together ALL findings with focus on THIS patient
- Reference their age, conditions, current medications
- Explain how everything applies to THEM specifically
- DO NOT use generic phrases like "commonly", "generally", "most people"
- Examples:
  * "For you specifically, Bharathi, at age 25 with no existing medications or health conditions, this looks very safe."
  * "Given your age of 72, your diabetes, and the fact you're already taking Metformin and Lisinopril, adding this medication requires caution because..."
  * "In your case, with your heart condition and current blood thinner, this combination is too risky because..."

SECTION 7: CLEAR RECOMMENDATION
- State clearly whether it's safe to add or not
- Give specific reasons based on ALL the data you analyzed
- Provide actionable advice
- Examples:
  * "This looks safe to add! The AI risk is low, no interactions were found, your dosage is appropriate, and given your age and health status, you should tolerate it well."
  * "I strongly advise against adding this. The high interaction risk, combined with your existing heart medication and the dosage concerns, make this too dangerous."

SECTION 8: FINAL VERDICT (REQUIRED!)
- End with a clear verdict on a new line
- Format: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD"

[COMPLETE EXAMPLE - SAFE CASE WITH ALL SECTIONS]

Hi Bharathi!

I've done a complete analysis of adding Acitretin to your medication plan.

My AI system calculated a 0% interaction risk - that's excellent news and very reassuring.

I searched my database thoroughly for interactions between Acitretin and any potential medications you might take in the future, and I found no documented interactions. This means Acitretin is generally safe to combine with other medications.

Your dosage of 10mg once daily is perfect. This is well within the safe range (maximum is 50mg daily), and it's actually a conservative starting dose which is smart.

For you specifically at age 25, Acitretin may cause some dry skin and chapped lips - these are common but manageable. Keep lip balm handy and use a good moisturizer. Since you're young and healthy with no existing conditions, your body should handle these minor side effects well.

Given your age, your current health status with no existing medications or conditions, and the complete absence of interaction risks, this medication looks very safe for you to start. The dosage is appropriate and conservative.

This looks safe to add! Just remember to follow the prescribed dosage of 10mg once daily, and stay hydrated to help with any dryness.

Verdict: SAFE TO ADD

[COMPLETE EXAMPLE - HIGH RISK CASE WITH ALL SECTIONS]

Hi Ganesh!

I've carefully analyzed adding Ibuprofen to your current medications, and I have some serious concerns.

My AI system shows a 92% interaction risk - that's very high and indicates a dangerous combination.

I found a critical interaction in my database: Warfarin and Ibuprofen interact severely. Here's what happens - Ibuprofen interferes with your blood's ability to clot, and when combined with Warfarin (which you're already taking as a blood thinner), this dramatically increases your risk of internal bleeding. This could lead to bleeding in your stomach, brain, or other organs, which can be life-threatening.

Your planned dosage of 400mg three times daily would normally be safe for Ibuprofen alone, but in your case, ANY amount of Ibuprofen is dangerous because of the Warfarin interaction.

For someone your age at 68, bleeding complications are even more serious. Your body doesn't heal as quickly, and internal bleeding can cause severe damage before you even notice symptoms. Additionally, with your heart condition, any bleeding incident could be catastrophic.

Given your age, your heart condition, the fact you're on Warfarin, and this severe documented interaction, I strongly advise against taking Ibuprofen. Please talk to your doctor about safer pain relief options like Acetaminophen (Tylenol), which doesn't interact with Warfarin.

Verdict: DO NOT ADD

[CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE]
- ALWAYS include GNN risk score if it's in the data
- ALWAYS explain pairwise interactions if found in database
- ALWAYS assess dosage safety if dosage info is provided
- ALWAYS mention relevant side effects if available
- ALWAYS personalize to patient's age, conditions, medications
- NEVER skip sections that have data available
- NEVER use generic phrases like "commonly", "generally", "most people"
- ALWAYS explain CONSEQUENCES (what actually happens), not just "there's a risk"
- ALWAYS end with clear verdict
- Weave all sections together naturally in conversational tone
"""
    
    # Try cloud API first (OpenRouter.ai), fallback to localhost/deterministic response
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if OPENROUTER_COOLDOWN_UNTIL and time.time() < OPENROUTER_COOLDOWN_UNTIL:
        return generate_fallback_response(context)
    
    if api_key:
        # Use a single configured OpenRouter model and fail fast on billing/model errors.
        # This avoids noisy repeated warnings for every request.
        print("[INFO] Using OpenRouter.ai cloud API")
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Medicine Assistant"
        }

        # Configure from env to avoid hardcoded outdated model names
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        try:
            print(f"[INFO] Trying model: {model}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 2000
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)

            # Fail fast on known non-retryable API states
            if response.status_code == 402:
                print("[WARNING] OpenRouter billing/credits unavailable (402). Using deterministic fallback.")
                OPENROUTER_COOLDOWN_UNTIL = time.time() + 1800  # 30 minutes
                OPENROUTER_COOLDOWN_REASON = "billing"
                return generate_fallback_response(context)
            if response.status_code in [400, 401, 403, 404]:
                print(f"[WARNING] OpenRouter request/model configuration issue ({response.status_code}). Using deterministic fallback.")
                OPENROUTER_COOLDOWN_UNTIL = time.time() + 900  # 15 minutes
                OPENROUTER_COOLDOWN_REASON = "config"
                return generate_fallback_response(context)

            response.raise_for_status()
            print(f"[SUCCESS] Model {model} responded successfully")
            return response.json()['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f"[WARNING] OpenRouter request failed ({model}): {e}. Using deterministic fallback.")
            OPENROUTER_COOLDOWN_UNTIL = time.time() + 300  # 5 minutes transient cooldown
            OPENROUTER_COOLDOWN_REASON = "network"
            return generate_fallback_response(context)
    else:
        # Fallback to local API (original behavior)
        print("[INFO] Using local LLM API")
        api_url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Local LLM API error: {e}")
            return generate_fallback_response(context)

# --- USER AUTHENTICATION & PROFILE (UPGRADED) ---
# ===== OLD HTML ROUTES - DISABLED FOR REACT FRONTEND =====
# These routes are commented out because we're using React for the frontend
# The API routes below handle all frontend requests

# @app.route('/')
# def home(): return redirect(url_for('login'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     ... (disabled)

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     ... (disabled)

# @app.route('/dashboard')
# def dashboard():
#     ... (disabled)

# @app.route('/profile', methods=['GET', 'POST'])
# def profile():
#     ... (disabled)

# ... (The rest of your app.py file, including check_before_adding, add_medication, etc., remains the same)
def get_holistic_context(new_drug, patient, existing_meds, dosage_amount=None, dosage_unit=None, frequency=None, openfda_dosage_text=None):
    patient_age = calculate_age(patient.get('dob'))
    patient['age'] = patient_age
    
    # Safely get patient data with defaults
    name = patient.get('name') or 'Patient'
    conditions = patient.get('conditions') or 'None reported'
    allergies = patient.get('drug_allergies') or 'None reported'
    
    context_str = f"Patient Profile: Name: {name}, Age: {patient_age}, Conditions: {conditions}, Allergies: {allergies}.\n"
    context_str += f"Current Medications: {', '.join([med['drug_name'] for med in existing_meds]) if existing_meds else 'None'}.\n"
    context_str += f"New Drug to Analyze: {new_drug}.\n\n"
    
    # Add GNN prediction if model is available
    if gnn_model and drug_map and existing_meds:
        gnn_risk = get_gnn_prediction(new_drug, existing_meds)
        context_str += f"GNN Predicted Risk: {gnn_risk:.1f}% chance of interaction\n\n"
    else:
        context_str += "GNN Predicted Risk: Unable to calculate (model not available or no existing medications)\n\n"
    
    # Add side effects information
    side_effects_info = side_effects_db.get_side_effects(new_drug, patient)
    if side_effects_info['side_effects']:
        context_str += f"Common Side Effects of {new_drug}:\n"
        for effect in side_effects_info['side_effects'][:3]:  # Limit to 3 most common
            context_str += f"- {effect}\n"
        context_str += "\n"
    
    if side_effects_info['risk_warnings']:
        context_str += f"Personalized Risk Warnings for {patient.get('name')}:\n"
        for warning in side_effects_info['risk_warnings']:
            context_str += f"- {warning}\n"
        context_str += "\n"
    
    # Add multi-drug conflict analysis
    triangular_conflicts = multi_drug_checker.check_triangular_conflicts(new_drug, existing_meds)
    category_conflicts = multi_drug_checker.check_category_conflicts(new_drug, existing_meds)
    
    if triangular_conflicts:
        context_str += "Multi-Drug Conflict Analysis (Triangular Conflicts):\n"
        for conflict in triangular_conflicts:
            context_str += f"- {conflict['warning']}\n"
            context_str += f"  Drugs involved: {', '.join(conflict['drugs'])}\n"
            context_str += f"  Severity: {conflict['severity']}\n"
        context_str += "\n"
    
    if category_conflicts:
        context_str += "Category-Based Conflict Analysis:\n"
        for conflict in category_conflicts:
            context_str += f"- {conflict['warning']}\n"
        context_str += "\n"
    
    # Add OpenFDA dosage comparison (unit-aware, no summary)
    if dosage_amount and openfda_dosage_text:
        comparison_message = compare_dosage(
            {
                "dosage_amount": dosage_amount,
                "dosage_unit": dosage_unit,
                "frequency": frequency
            },
            openfda_dosage_text
        )
        
        context_str += f"Dosage Comparison with FDA Label:\n"
        context_str += f"- User entered dose: {dosage_amount} {dosage_unit or 'mg'} {frequency or ''}\n"
        context_str += f"- FDA label text available: yes\n"
        context_str += f"- Comparison result: {comparison_message}\n"
        context_str += "\n"
    
    context_str += "Known Pairwise Interactions from Database (for reference):\n"
    found_pairwise = False
    for med in existing_meds:
        interaction = rag_system.search_interaction(new_drug, med['drug_name'])
        if interaction:
            found_pairwise = True
            context_str += f"- {interaction['drug_a']} and {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}\n"
    if not found_pairwise: 
        context_str += "- No specific pairwise interactions were found in the knowledge base.\n"
    
    return context_str

def get_gnn_prediction(new_drug, existing_meds):
    """Get GNN prediction for drug interactions"""
    try:
        if not gnn_model or not drug_map:
            return 0.0
        
        # Convert drug names to indices
        new_drug_idx = None
        existing_drug_indices = []
        
        # Find new drug index
        for drug_name, idx in drug_map.items():
            if new_drug.lower() in drug_name.lower() or drug_name.lower() in new_drug.lower():
                new_drug_idx = idx
                break
        
        if new_drug_idx is None:
            return 0.0
        
        # Find existing drug indices
        for med in existing_meds:
            for drug_name, idx in drug_map.items():
                if med['drug_name'].lower() in drug_name.lower() or drug_name.lower() in med['drug_name'].lower():
                    existing_drug_indices.append(idx)
                    break
        
        if not existing_drug_indices:
            return 0.0
        
        # Calculate average risk across all existing medications
        total_risk = 0.0
        count = 0
        
        for existing_idx in existing_drug_indices:
            # Create edge index for this pair
            edge_index = torch.tensor([[new_drug_idx], [existing_idx]], dtype=torch.long)
            
            # Get node embeddings
            with torch.no_grad():
                # Create a simple edge index for the model
                all_nodes = torch.tensor(list(range(len(drug_map))), dtype=torch.long)
                edge_index_full = torch.tensor([[new_drug_idx, existing_idx], [existing_idx, new_drug_idx]], dtype=torch.long)
                
                # Get predictions
                z = gnn_model.encode(all_nodes, edge_index_full)
                pred = gnn_model.decode(z, edge_index)
                risk_score = torch.sigmoid(pred).item()
                
                total_risk += risk_score
                count += 1
        
        if count > 0:
            avg_risk = (total_risk / count) * 100
            return min(avg_risk, 100.0)  # Cap at 100%
        
        return 0.0
        
    except Exception as e:
        print(f"Error in GNN prediction: {e}")
        return 0.0

def _row_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        return default

def collect_interaction_evidence(new_drug, existing_meds):
    pairwise_found = []
    med_names = []
    for med in (existing_meds or []):
        med_name = _row_get(med, 'drug_name', '')
        if med_name:
            med_names.append(med_name)
            if rag_system and getattr(rag_system, "df", None) is not None:
                interaction = rag_system.search_interaction(new_drug, med_name)
                if interaction:
                    pairwise_found.append(interaction)

    mock_meds = [{'drug_name': name} for name in med_names]
    triangular_conflicts = multi_drug_checker.check_triangular_conflicts(new_drug, mock_meds)
    category_conflicts = multi_drug_checker.check_category_conflicts(new_drug, mock_meds)

    return {
        "pairwise": pairwise_found,
        "triangular": triangular_conflicts,
        "category": category_conflicts,
        "rag_available": rag_system is not None and getattr(rag_system, "df", None) is not None,
        "med_count": len(med_names)
    }

def calculate_evidence_based_risk(evidence, gnn_risk=0.0):
    interaction_count = len(evidence.get("pairwise", [])) + len(evidence.get("triangular", [])) + len(evidence.get("category", []))
    rag_available = evidence.get("rag_available", False)

    if not rag_available:
        return {"status": "UNKNOWN", "score": None, "display": "Insufficient Data"}

    if interaction_count > 0:
        score = min(100.0, 70.0 + interaction_count * 8.0 + (gnn_risk * 0.1))
        return {"status": "HIGH", "score": round(score, 1), "display": f"{round(score, 1)}%"}

    score = max(0.0, min(20.0, gnn_risk * 0.2))
    return {"status": "LOW", "score": round(score, 1), "display": f"{round(score, 1)}%"}

def build_safety_response(
    patient,
    new_drug,
    evidence,
    risk_result,
    dosage_message,
    openfda_text,
    dosage_source,
    rule_layer,
    dosage_validation
):
    patient_name = (patient or {}).get('name') or "Patient"
    pairwise = evidence.get("pairwise", [])
    triangular = evidence.get("triangular", [])
    category = evidence.get("category", [])
    med_count = evidence.get("med_count", 0)

    side_effects_info = side_effects_db.get_side_effects(new_drug, patient or {})
    common_side_effects = side_effects_info.get('side_effects') or []
    serious_effects = side_effects_info.get('serious_effects') or []

    def _plain_language_fda_summary(fda_text):
        if not fda_text:
            return ["FDA label dosage details are not available."]
        text = fda_text.replace("\n", " ")
        summary = []
        dose_matches = re.findall(r'(\d+(?:\.\d+)?)\s*mg(?:/day| daily)?', text, re.IGNORECASE)
        if dose_matches:
            unique_vals = []
            for d in dose_matches:
                if d not in unique_vals:
                    unique_vals.append(d)
            preview = ", ".join(unique_vals[:4])
            summary.append(f"FDA label mentions common dose values such as: {preview} mg.")
        if re.search(r'with meals', text, re.IGNORECASE):
            summary.append("FDA label says this medicine should be taken with meals.")
        if re.search(r'gradual|increase slowly|weekly intervals', text, re.IGNORECASE):
            summary.append("Dose is usually started low and increased gradually.")
        if re.search(r'maximum|do not exceed', text, re.IGNORECASE):
            summary.append("FDA label includes maximum daily limits; follow those limits carefully.")
        if not summary:
            summary.append("FDA dosage instructions are complex. Please read the full text below and confirm with your doctor.")
        return summary

    lines = []
    lines.append("Intro")
    lines.append(f"Hi {patient_name}, this analysis is based on available medication databases and your provided profile.")
    lines.append("")
    lines.append("Interaction Analysis (Data-Based)")
    if pairwise or triangular or category:
        lines.append(f"Documented interaction signals were found for {new_drug}.")
        for item in pairwise[:3]:
            lines.append(f"- {item.get('drug_a')} + {item.get('drug_b')} ({item.get('severity', 'unknown')}): {item.get('interaction', 'Documented interaction')}")
        for item in triangular[:2]:
            lines.append(f"- Multi-drug conflict: {', '.join(item.get('drugs', []))} ({item.get('severity', 'unknown')})")
        for item in category[:2]:
            lines.append(f"- Category conflict: {item.get('warning', item.get('description', 'Conflict noted'))}")
    elif med_count == 0:
        lines.append("No current medications are saved, so pairwise interaction checking cannot be performed yet.")
    else:
        lines.append("No interactions were found in available databases for this combination.")

    lines.append(f"Risk score: {risk_result['display']}")
    lines.append(f"GNN model risk analysis: {risk_result['display']} predicted interaction signal.")
    lines.append("")
    lines.append("Dosage Comparison")
    lines.append(dosage_message)
    lines.append("")
    lines.append("Side Effects")
    if common_side_effects:
        lines.append(f"Common effects: {', '.join(common_side_effects[:4])}.")
    else:
        lines.append("No specific side effects were found for this drug in the local side-effects dataset.")
    if serious_effects:
        lines.append(f"Serious effects to watch: {', '.join(serious_effects[:2])}.")
    lines.append("")
    lines.append("Recommendation")
    if risk_result["status"] == "HIGH" or rule_layer.get("hard_block"):
        lines.append("Potential interaction risk is present. This does not guarantee harm, but caution is required.")
    elif risk_result["status"] == "LOW":
        lines.append("No interactions were found in available databases. This does not guarantee absence of risk.")
    else:
        lines.append("Risk could not be fully determined from available data. This does not guarantee absence of risk.")
    if dosage_validation and dosage_validation.get("warnings"):
        lines.append("Dosage warnings are present and should be reviewed carefully.")
    lines.append("Consult a healthcare professional before making medication changes.")
    lines.append("")
    lines.append("Clinical Safety Layer")
    lines.append(rule_layer.get("text", "Clinical safety layer not available."))
    lines.append("")
    lines.append("Dosage Information (OpenFDA FULL TEXT)")
    lines.append("Plain-language FDA summary:")
    for s in _plain_language_fda_summary(openfda_text):
        lines.append(f"- {s}")
    lines.append("Dosage Information")
    lines.append(openfda_text if openfda_text else FDA_DOSAGE_MISSING_TEXT)
    if dosage_source == "OpenFDA":
        lines.append("Source: OpenFDA")
    elif dosage_source == "DrugBank":
        lines.append("Source: DrugBank (OpenFDA not available)")
    else:
        lines.append("Source: OpenFDA (not available)")
    return "\n".join(lines)

@app.route('/check_before_adding', methods=['POST'])
def check_before_adding():
    try:
        if 'patient_id' not in session: 
            return jsonify({'error': 'User not logged in'}), 401
        
        data = request.json
        if not data or 'drug_name' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        new_drug = data['drug_name']
        dosage_amount = data.get('dosage_amount', '')
        dosage_unit = data.get('dosage_unit', '')
        frequency = data.get('frequency', '')
        
        # Always fetch OpenFDA dosage text for display section
        openfda_dosage_text = get_cached_dosage(new_drug)
        if not openfda_dosage_text:
            openfda_dosage_text = fetch_openfda_dosage(new_drug)
            if openfda_dosage_text:
                cache_dosage(new_drug, openfda_dosage_text)
        # If OpenFDA is unavailable, fall back to local dosage database so user still gets guidance.
        drugbank_fallback = None
        if not openfda_dosage_text:
            drugbank_fallback = check_drugbank_dosage(new_drug)

        dosage_info = {
            "dosage": (
                openfda_dosage_text
                if openfda_dosage_text
                else (drugbank_fallback if drugbank_fallback else FDA_DOSAGE_MISSING_TEXT)
            ),
            "dosage_source": "OpenFDA" if openfda_dosage_text else ("DrugBank" if drugbank_fallback else "None")
        }
        
        # Validate dosage first
        dosage_validation = dosage_validator.validate_dosage(new_drug, dosage_amount, dosage_unit, frequency)
        
        conn = get_db_connection()
        patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        if not patient_data:
            conn.close()
            return jsonify({'error': 'Patient not found'}), 404
            
        patient = dict(patient_data)
        existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
        conn.close()
        
        # Get GNN risk score
        gnn_risk = 0.0
        if gnn_model and drug_map and existing_meds:
            gnn_risk = get_gnn_prediction(new_drug, existing_meds)

        evidence = collect_interaction_evidence(new_drug, existing_meds)
        risk_result = calculate_evidence_based_risk(evidence, gnn_risk)
        dosage_message = compare_dosage(
            {
                "dosage_amount": dosage_amount,
                "dosage_unit": dosage_unit,
                "frequency": frequency
            },
            openfda_dosage_text
        )

        # Add deterministic medical realism layer (side effects, multi-drug conflicts, smart dosage suggestion)
        rule_layer = build_rule_based_safety_layer(
            new_drug=new_drug,
            patient=patient,
            existing_meds=existing_meds,
            gnn_risk=gnn_risk,
            dosage_validation=dosage_validation,
            dosage_amount=dosage_amount,
            dosage_unit=dosage_unit,
            frequency=frequency
        )

        if risk_result["status"] == "HIGH" or rule_layer.get("hard_block"):
            verdict = "DO NOT ADD"
            can_add = False
        elif risk_result["status"] == "UNKNOWN" or rule_layer.get("caution") or (dosage_validation and not dosage_validation.get("is_safe", True)):
            verdict = "CAUTION ADVISED"
            can_add = False
        else:
            verdict = "SAFE TO ADD"
            can_add = True

        main_summary = build_safety_response(
            patient=patient,
            new_drug=new_drug,
            evidence=evidence,
            risk_result=risk_result,
            dosage_message=dosage_message,
            openfda_text=openfda_dosage_text if openfda_dosage_text else (drugbank_fallback if drugbank_fallback else None),
            dosage_source=dosage_info["dosage_source"],
            rule_layer=rule_layer,
            dosage_validation=dosage_validation
        )
        
        # Format response for React frontend
        response_data = {
            'gnn_risk': risk_result['score'] if risk_result['score'] is not None else 0,
            'risk_status': risk_result['status'],
            'risk_display': risk_result['display'],
            'verdict': verdict,
            'ai_response': main_summary,
            'can_add': can_add,
            'dosage_validation': dosage_validation,
            'dosage': dosage_info['dosage'],
            'dosage_source': dosage_info['dosage_source']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in check_before_adding: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'gnn_risk': 0,
            'verdict': 'ERROR',
            'ai_response': 'I apologize, but there was an error processing your request. Please try again.',
            'can_add': False,
            'dosage_validation': {'is_safe': False, 'warnings': ['Error occurred during validation']},
            'dosage': FDA_DOSAGE_MISSING_TEXT,
            'dosage_source': 'None'
        }), 500

@app.route('/add_medication', methods=['POST'])
def add_medication():
    if 'patient_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json() if request.is_json else request.form
    form_data = (
        session['patient_id'],
        data.get('drug_name'),
        data.get('dosage_amount'),
        data.get('dosage_unit'),
        data.get('frequency'),
        data.get('start_date'),
        data.get('end_date')
    )
    
    conn = get_db_connection()
    conn.execute('INSERT INTO medications (patient_id, drug_name, dosage_amount, dosage_unit, frequency, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)', form_data)
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': f"'{data.get('drug_name')}' has been added to your log."})

@app.route('/api/medications/<int:medication_id>', methods=['DELETE'])
def delete_medication(medication_id):
    """Delete a medication from the patient's profile"""
    if 'patient_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        conn = get_db_connection()
        
        # Verify the medication belongs to the current patient
        medication = conn.execute(
            'SELECT * FROM medications WHERE id = ? AND patient_id = ?',
            (medication_id, session['patient_id'])
        ).fetchone()
        
        if not medication:
            conn.close()
            return jsonify({'success': False, 'error': 'Medication not found or unauthorized'}), 404
        
        # Delete the medication
        conn.execute('DELETE FROM medications WHERE id = ? AND patient_id = ?', 
                    (medication_id, session['patient_id']))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f"Medication '{medication['drug_name']}' has been removed successfully."
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to delete medication: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete medication'}), 500

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    """AI Assistant endpoint with intent classification"""
    if 'patient_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        message = data.get('question', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Step 1: Classify intent
        intent = intent_classifier.classify(message)
        
        # Step 2: Handle based on intent type
        if intent.type == "conversational":
            # Use ConversationalHandler for casual conversation
            response = conversational_handler.handle(message, intent)
            
            response_data = {
                'response': response,
                'intent': 'conversational',
                'timestamp': datetime.now().isoformat()
            }
            
            # Validate response format
            if not validate_response_format(response_data, 'chatbot_conversational'):
                print("[ERROR] Chatbot: Conversational response format validation failed")
                return jsonify({'error': 'Internal validation error'}), 500
            
            return jsonify(response_data)
        
        # Step 3: Medical intent - use InteractionEngine
        conn = get_db_connection()
        patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        patient = dict(patient_data)
        existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
        conn.close()
        
        # Extract drug from message or use existing meds
        if intent.extracted_drugs:
            new_drug = intent.extracted_drugs[0]
            existing_drugs = [med['drug_name'] for med in existing_meds]
        else:
            # Fallback: try to extract drug name from message
            new_drug_match = re.search(r'(take|about|check|add)\s+([\w\s-]+)\??', message.lower())
            if new_drug_match:
                new_drug = new_drug_match.group(2).strip()
            else:
                new_drug = message
            existing_drugs = [med['drug_name'] for med in existing_meds]
        
        # Use Interaction Engine for medical analysis
        result = interaction_engine.analyze_interaction(
            new_drug=new_drug,
            existing_drugs=existing_drugs,
            patient_profile=patient
        )
        
        # Validate InteractionResult format
        if not validate_interaction_result(result):
            print("[ERROR] Chatbot: InteractionResult validation failed")
            return jsonify({'error': 'Internal validation error'}), 500
        
        response_data = {
            'response': result.llm_explanation,
            'verdict': result.verdict,
            'gnn_risk': result.gnn_risk,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate response format
        if not validate_response_format(response_data, 'chatbot_medical'):
            print("[ERROR] Chatbot: Medical response format validation failed")
            return jsonify({'error': 'Internal validation error'}), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process message'}), 500

@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    return jsonify({'success': True, 'message': 'You have been logged out.'})

# --- Live Health Monitoring Endpoints ---
@app.route('/api/health-data')
def get_health_data():
    """Get live health data (simulated for demo)"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # Simulate live health data
    import random
    from datetime import datetime, timedelta
    
    # Generate realistic health data
    heart_rate = random.randint(65, 85)
    steps = random.randint(8000, 12000)
    calories = random.randint(200, 400)
    
    # Generate 7-day trend data
    trend_data = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        trend_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'heart_rate': random.randint(70, 80),
            'steps': random.randint(7000, 11000),
            'calories': random.randint(180, 350)
        })
    
    return jsonify({
        'current': {
            'heart_rate': heart_rate,
            'steps': steps,
            'calories': calories,
            'timestamp': datetime.now().isoformat()
        },
        'trends': trend_data,
        'status': 'connected'
    })

@app.route('/api/google-fit-auth')
def google_fit_auth():
    """Initiate Google Fit OAuth flow"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # In a real implementation, this would redirect to Google OAuth
    # For demo purposes, return instructions
    return jsonify({
        'message': 'Google Fit integration requires OAuth setup',
        'instructions': [
            '1. Install Google Fit app on your phone',
            '2. Enable permissions for step counting and heart rate',
            '3. The app will automatically sync data',
            '4. Currently showing simulated data for demo'
        ],
        'demo_mode': True
    })

@app.route('/api/health-alerts')
def get_health_alerts():
    """Get health alerts based on current data"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    conn = get_db_connection()
    patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    
    alerts = []
    
    # Check for medication-related alerts
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    
    # Example alerts based on patient profile
    if patient_data['is_smoker'] == 'Yes':
        alerts.append({
            'type': 'warning',
            'title': 'Smoking Alert',
            'message': 'Smoking can affect medication effectiveness. Consider discussing with your doctor.',
            'icon': '🚭'
        })
    
    if patient_data['weight_kg'] and float(patient_data['weight_kg']) > 100:
        alerts.append({
            'type': 'info',
            'title': 'Weight Consideration',
            'message': 'Your weight may affect medication dosing. Monitor for any unusual effects.',
            'icon': '⚖️'
        })
    
    return jsonify({
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    })

# --- Emergency Mode ---
@app.route('/emergency-check', methods=['POST'])
def emergency_check():
    """Emergency check using unified Interaction Engine"""
    try:
        data = request.get_json(silent=True)
        print(f"[DEBUG] Emergency check received: {data}")
        
        if not data:
            return jsonify({'status': 'UNSAFE', 'reason': 'No data provided', 'error': 'Invalid request'}), 400
        
        if 'drug1' not in data or 'drug2' not in data:
            return jsonify({'status': 'UNSAFE', 'reason': 'Please provide two drug names', 'error': 'Missing drug names'}), 400
        
        drug1 = data['drug1'].strip()
        drug2 = data['drug2'].strip()
        
        if not drug1 or not drug2:
            return jsonify({'status': 'UNSAFE', 'reason': 'Both drug names are required', 'error': 'Empty drug names'}), 400
        
        print(f"[DEBUG] Checking interaction between: {drug1} and {drug2}")
        
        # Use Interaction Engine for unified analysis
        result = interaction_engine.analyze_interaction(
            new_drug=drug1,
            existing_drugs=[drug2],
            patient_profile=None  # Anonymous emergency check
        )
        
        # Validate InteractionResult format
        if not validate_interaction_result(result):
            print("[ERROR] Emergency Check: InteractionResult validation failed")
            return jsonify({
                'status': 'UNSAFE',
                'response': 'Internal validation error. Please consult a healthcare professional.',
                'error': 'Validation failed'
            }), 500
        
        # Map verdict to status
        status_map = {
            'SAFE TO ADD': 'SAFE',
            'CAUTION ADVISED': 'CAUTION',
            'DO NOT ADD': 'UNSAFE'
        }
        status = status_map.get(result.verdict, 'UNSAFE')
        
        print(f"[DEBUG] Status: {status}, GNN Risk: {result.gnn_risk}%, Verdict: {result.verdict}")
        
        response_data = {
            'status': status,
            'response': result.llm_explanation,
            'gnn_risk': result.gnn_risk,
            'drug1': drug1,
            'drug2': drug2,
            'interaction': result.rag_interactions[0] if result.rag_interactions else None,
            'timestamp': result.timestamp
        }
        
        # Validate response format
        if not validate_response_format(response_data, 'emergency_check'):
            print("[ERROR] Emergency Check: Response format validation failed")
            return jsonify({
                'status': 'UNSAFE',
                'response': 'Internal validation error. Please consult a healthcare professional.',
                'error': 'Validation failed'
            }), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Exception in emergency check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'UNSAFE',
            'response': 'Unable to perform emergency check. Please consult a healthcare professional immediately.',
            'error': str(e)
        }), 500

# --- Google Fit OAuth2 Integration ---
# Configuration (you'll need to set these as environment variables)
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:5000/oauth2callback')

SCOPES = ['https://www.googleapis.com/auth/fitness.heart_rate.read',
          'https://www.googleapis.com/auth/fitness.activity.read',
          'https://www.googleapis.com/auth/fitness.body.read']

def credentials_to_dict(credentials):
    """Convert credentials object to dictionary for storage in session"""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@app.route('/authorize-fit')
def authorize_fit():
    """Initiate Google Fit OAuth2 flow"""
    print("[DEBUG] Starting Google Fit authorization")
    
    if not GOOGLE_FIT_AVAILABLE:
        return jsonify({
            'error': 'Google Fit not available',
            'message': 'Please install: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client'
        }), 500
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({
            'error': 'Google Fit not configured',
            'message': 'Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables'
        }), 500
    
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [GOOGLE_REDIRECT_URI]
                }
            },
            scopes=SCOPES
        )
        
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        session['state'] = state
        print(f"[DEBUG] Redirecting to: {authorization_url}")
        
        return redirect(authorization_url)
        
    except Exception as e:
        print(f"[ERROR] Authorization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/oauth2callback')
def oauth2callback():
    """Handle Google OAuth2 callback"""
    print("[DEBUG] OAuth2 callback received")
    
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [GOOGLE_REDIRECT_URI]
                }
            },
            scopes=SCOPES,
            state=session['state']
        )
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        
        authorization_response = request.url
        flow.fetch_token(authorization_response=authorization_response)
        
        # Store credentials in session
        session['credentials'] = credentials_to_dict(flow.credentials)
        print("[DEBUG] Credentials stored in session")
        
        flash('Successfully connected to Google Fit!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"[ERROR] Callback error: {e}")
        flash('Failed to connect to Google Fit. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/disconnect-fit')
def disconnect_fit():
    """Disconnect Google Fit"""
    session.pop('credentials', None)
    flash('Disconnected from Google Fit', 'info')
    return redirect(url_for('dashboard'))

@app.route('/get-fit-data')
def get_fit_data():
    """Fetch real-time health data from Google Fit API"""
    if 'patient_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # Check if Google Fit is connected
    if 'credentials' not in session:
        # Return fallback data if not connected
        return get_dummy_health_data()
    
    try:
        # Get credentials from session
        creds_dict = session['credentials']
        
        # Build credentials object
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=creds_dict.get('token'),
            refresh_token=creds_dict.get('refresh_token'),
            token_uri=creds_dict.get('token_uri'),
            client_id=creds_dict.get('client_id'),
            client_secret=creds_dict.get('client_secret'),
            scopes=creds_dict.get('scopes')
        )
        
        # Refresh if necessary
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            session['credentials'] = credentials_to_dict(creds)
        
        # Fetch health data
        health_data = fetch_google_fit_data(creds)
        
        return jsonify({
            'status': 'connected',
            'data_source': 'google_fit',
            'current': health_data['current'],
            'trends': health_data['trends'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Error fetching Google Fit data: {e}")
        return get_dummy_health_data()

def fetch_google_fit_data(credentials):
    """Fetch data from Google Fit API"""
    try:
        import googleapiclient.discovery
        
        fit_service = googleapiclient.discovery.build('fitness', 'v1', credentials=credentials)
        
        # Define time range (last 24 hours)
        end_time = int(datetime.now().timestamp() * 1000000000)  # nanoseconds
        start_time = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000000000)
        
        # Get heart rate
        heart_rate_data = fit_service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm',
            datasetId=f'{start_time}-{end_time}'
        ).execute()
        
        # Get steps
        steps_data = fit_service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.step_count.delta:com.google.android.gms:estimated_steps',
            datasetId=f'{start_time}-{end_time}'
        ).execute()
        
        # Process heart rate
        heart_rate = 75  # default
        if 'point' in heart_rate_data and len(heart_rate_data['point']) > 0:
            latest_point = heart_rate_data['point'][-1]
            heart_rate = int(latest_point['value'][0]['fpVal'])
        
        # Process steps
        steps = 0
        if 'point' in steps_data:
            for point in steps_data['point']:
                steps += int(point['value'][0]['intVal'])
        
        # Get calories (estimated)
        calories = int(steps * 0.05)  # Rough estimate: 0.05 cal per step
        
        # Generate trend data (last 7 days)
        trends = []
        for i in range(7):
            trends.append({
                'date': (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d'),
                'heart_rate': heart_rate + random.randint(-5, 5),
                'steps': max(0, steps + random.randint(-1000, 1000)),
                'calories': int((steps + random.randint(-1000, 1000)) * 0.05)
            })
        
        return {
            'current': {
                'heart_rate': heart_rate,
                'steps': steps,
                'calories': calories
            },
            'trends': trends
        }
        
    except Exception as e:
        print(f"[ERROR] Error in fetch_google_fit_data: {e}")
        return get_dummy_health_data()

def get_dummy_health_data():
    """Generate dummy health data as fallback"""
    heart_rate = random.randint(65, 85)
    steps = random.randint(8000, 12000)
    calories = random.randint(200, 400)
    
    trends = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        trends.append({
            'date': date.strftime('%Y-%m-%d'),
            'heart_rate': random.randint(70, 80),
            'steps': random.randint(7000, 11000),
            'calories': random.randint(180, 350)
        })
    
    return jsonify({
        'status': 'simulated',
        'data_source': 'dummy',
        'current': {
            'heart_rate': heart_rate,
            'steps': steps,
            'calories': calories,
            'timestamp': datetime.now().isoformat()
        },
        'trends': trends
    })

@app.route('/api/google-fit-connect')
def google_fit_connect():
    """Initiate Google Fit connection"""
    return redirect(url_for('authorize_fit'))

# ===== API ROUTES FOR REACT FRONTEND =====

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'patient_id' in session:
        conn = get_db_connection()
        patient = conn.execute('SELECT id, name, email FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
        conn.close()
        if patient:
            return jsonify({
                'authenticated': True,
                'user': {'id': patient['id'], 'name': patient['name'], 'email': patient['email']}
            })
    return jsonify({'authenticated': False})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if patient and check_password_hash(patient['password_hash'], password):
        session['patient_id'] = patient['id']
        return jsonify({
            'success': True,
            'user': {'id': patient['id'], 'name': patient['name'], 'email': patient['email']}
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not is_valid_email(email):
        return jsonify({'success': False, 'error': 'Invalid email address'}), 400
    
    is_strong, message = is_strong_password(password)
    if not is_strong:
        return jsonify({'success': False, 'error': message}), 400
    
    conn = get_db_connection()
    if conn.execute('SELECT id FROM patients WHERE email = ?', (email,)).fetchone():
        conn.close()
        return jsonify({'success': False, 'error': 'Email already exists'}), 400
    
    password_hash = generate_password_hash(password)
    conn.execute('INSERT INTO patients (name, email, password_hash) VALUES (?, ?, ?)', (name, email, password_hash))
    conn.commit()
    new_patient = conn.execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    session['patient_id'] = new_patient['id']
    return jsonify({
        'success': True,
        'user': {'id': new_patient['id'], 'name': new_patient['name'], 'email': new_patient['email']}
    })

@app.route('/api/logout', methods=['GET'])
def api_logout():
    session.pop('patient_id', None)
    return jsonify({'success': True})

@app.route('/api/health-data', methods=['GET'])
def api_health_data():
    if 'patient_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Mock health data - replace with real Google Fit data
    current_data = {
        'heart_rate': random.randint(60, 100),
        'steps': random.randint(3000, 15000),
        'calories': random.randint(1500, 3000)
    }
    
    # Generate 7 days of trend data
    trends = []
    for i in range(7):
        day = datetime.now() - timedelta(days=6-i)
        trends.append({
            'date': day.isoformat(),
            'heart_rate': random.randint(60, 100),
            'steps': random.randint(3000, 15000),
            'calories': random.randint(1500, 3000)
        })
    
    return jsonify({'current': current_data, 'trends': trends})

@app.route('/api/medications', methods=['GET'])
def api_medications():
    if 'patient_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = get_db_connection()
    medications = conn.execute('SELECT * FROM medications WHERE patient_id = ?', (session['patient_id'],)).fetchall()
    conn.close()
    
    meds_list = [dict(med) for med in medications]
    return jsonify({'medications': meds_list})

@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    if 'patient_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = get_db_connection()
    
    if request.method == 'POST':
        data = request.get_json()
        conn.execute('''UPDATE patients SET 
            name=?, dob=?, gender=?, weight_kg=?, height_cm=?, 
            emergency_contact=?, conditions=?, drug_allergies=?, 
            food_allergies=?, other_allergies=?, is_smoker=?, alcohol_consumption=? 
            WHERE id=?''', (
            data.get('name'), data.get('dob'), data.get('gender'),
            data.get('weight_kg'), data.get('height_cm'),
            data.get('emergency_contact'), data.get('conditions'),
            data.get('drug_allergies'), data.get('food_allergies'),
            data.get('other_allergies'), data.get('is_smoker'),
            data.get('alcohol_consumption'), session['patient_id']
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    
    # GET request
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (session['patient_id'],)).fetchone()
    conn.close()
    return jsonify({'profile': dict(patient)})

# --- Drug Search API ---
@app.route('/api/search-drugs', methods=['GET'])
def search_drugs():
    """Search for drugs by name (autocomplete)"""
    query = request.args.get('q', '').lower().strip()
    
    if len(query) < 2:
        return jsonify({'drugs': []})
    
    # Get unique drugs from the interactions database
    if rag_system.df is not None:
        all_drugs = set()
        all_drugs.update(rag_system.df['drug_a'].str.strip().tolist())
        all_drugs.update(rag_system.df['drug_b'].str.strip().tolist())
        
        # Filter drugs that match the query
        matching_drugs = [drug for drug in all_drugs if query in drug.lower()]
        matching_drugs = sorted(matching_drugs)[:20]  # Limit to 20 results
        
        return jsonify({'drugs': matching_drugs})
    
    return jsonify({'drugs': []})

# --- Quick Check API (No Login Required) ---
@app.route('/api/quick-check', methods=['POST'])
def quick_check():
    """Quick interaction check using unified Interaction Engine"""
    data = request.json
    drugs = data.get('drugs', [])
    use_profile = data.get('use_profile', False)
    
    if len(drugs) < 1:
        return jsonify({'error': 'At least one drug is required'}), 400
    
    # If use_profile is True, check if user is logged in
    if use_profile and 'patient_id' not in session:
        return jsonify({'error': 'Login required for profile-based checks'}), 401
    
    # Single drug case - use InteractionEngine with no existing drugs
    if len(drugs) == 1:
        result = interaction_engine.analyze_interaction(
            new_drug=drugs[0],
            existing_drugs=[],
            patient_profile=None  # Anonymous user
        )
        
        # Validate InteractionResult format
        if not validate_interaction_result(result):
            print("[ERROR] Quick Check: InteractionResult validation failed")
            return jsonify({'error': 'Internal validation error'}), 500
        
        response_data = {
            'gnn_risk': result.gnn_risk,
            'verdict': result.verdict,
            'ai_response': result.llm_explanation,
            'can_add': result.can_add,
            'dosage_validation': result.dosage_validation
        }
        
        # Validate response format
        if not validate_response_format(response_data, 'quick_check'):
            print("[ERROR] Quick Check: Response format validation failed")
            return jsonify({'error': 'Internal validation error'}), 500
        
        return jsonify(response_data)
    
    # Multiple drugs - use InteractionEngine
    # Treat first drug as "new drug" and rest as "existing drugs"
    new_drug = drugs[0]
    existing_drugs = drugs[1:]
    
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None  # Anonymous user
    )
    
    # Validate InteractionResult format
    if not validate_interaction_result(result):
        print("[ERROR] Quick Check: InteractionResult validation failed")
        return jsonify({'error': 'Internal validation error'}), 500
    
    # Map InteractionResult to existing response format for backward compatibility
    response_data = {
        'gnn_risk': result.gnn_risk,
        'verdict': result.verdict,
        'ai_response': result.llm_explanation,
        'can_add': result.can_add,
        'interactions': result.rag_interactions,
        'dosage_validation': result.dosage_validation
    }
    
    # Validate response format
    if not validate_response_format(response_data, 'quick_check'):
        print("[ERROR] Quick Check: Response format validation failed")
        return jsonify({'error': 'Internal validation error'}), 500
    
    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
