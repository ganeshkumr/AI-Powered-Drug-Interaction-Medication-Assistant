from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import requests
import json
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import date, datetime, timedelta
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os
import random
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
app.secret_key = 'the_final_and_most_secure_key'
CORS(app, supports_credentials=True, origins=['http://localhost:5173']) 

# --- GNN Model Definition and Loading ---
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
    try:
        with open('models/drug_map.json', 'r') as f: drug_map = json.load(f)
        model = GNNLinkPredictor(num_nodes=len(drug_map), embedding_dim=128, hidden_channels=128, out_channels=128)
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load('models/gnn_model.pt', map_location=map_location))
        model.eval(); print("[INFO] GNN Prediction Model loaded successfully on CPU.")
        return model, drug_map
    except FileNotFoundError:
        print("[ERROR] GNN model not found. Please run train_gnn.py first.")
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
rag_system = RAGSystem('interactions.csv')

# --- Drug Dosage Validation System ---
class DosageValidator:
    def __init__(self, dosage_file):
        try:
            with open(dosage_file, 'r') as f:
                self.dosage_limits = json.load(f)
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

dosage_validator = DosageValidator('drug_dosage_limits.json')

# --- Side Effects Database ---
class SideEffectsDatabase:
    def __init__(self, side_effects_file):
        try:
            with open(side_effects_file, 'r') as f:
                self.side_effects = json.load(f)
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
        elif risk_factor == "alcohol_use":
            alcohol = (patient_profile.get('alcohol_consumption') or '').lower()
            return alcohol in ['regular', 'occasional']
        return False

side_effects_db = SideEffectsDatabase('side_effects_database.json')

# --- Multi-Drug Conflict Checker ---
class MultiDrugConflictChecker:
    def __init__(self, conflicts_file):
        try:
            with open(conflicts_file, 'r') as f:
                self.conflicts_data = json.load(f)
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

multi_drug_checker = MultiDrugConflictChecker('multi_drug_conflicts.json')

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
        Analyze drug interactions using GNN + RAG + LLM
        
        Args:
            new_drug: The drug being added
            existing_drugs: List of existing drug names
            patient_profile: Optional patient information (age, conditions, etc.)
            dosage_info: Optional dosage details (amount, unit, frequency)
            
        Returns:
            InteractionResult containing all analysis results
        """
        try:
            # Handle single drug case (no existing drugs)
            if not existing_drugs or len(existing_drugs) == 0:
                return self._handle_single_drug(new_drug, patient_profile, dosage_info)
            
            # Step 1: GNN Prediction
            gnn_risk = self._get_gnn_prediction(new_drug, existing_drugs)
            
            # Step 2: RAG System Lookup
            rag_interactions = self._get_rag_interactions(new_drug, existing_drugs)
            
            # Step 3: Build comprehensive context
            context = self._build_context(
                new_drug, 
                existing_drugs, 
                gnn_risk, 
                rag_interactions,
                patient_profile,
                dosage_info
            )
            
            # Step 4: Get LLM explanation
            llm_explanation = self._get_llm_explanation(context)
            
            # Step 5: Determine verdict
            verdict = self._determine_verdict(llm_explanation, gnn_risk, rag_interactions)
            
            # Step 6: Validate dosage if provided
            dosage_validation = self._validate_dosage(new_drug, dosage_info)
            
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
            print(f"[ERROR] InteractionEngine error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback
            return InteractionResult(
                gnn_risk=0.0,
                rag_interactions=[],
                llm_explanation=f"Unable to analyze interaction. Please consult a healthcare professional. Error: {str(e)}",
                verdict="DO NOT ADD",
                can_add=False,
                dosage_validation={'is_safe': False, 'warnings': ['Analysis failed']},
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
        """Get GNN risk prediction for drug interactions"""
        try:
            if not self.gnn_model or not self.drug_map:
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
                return 0.0
            
            # Find existing drug indices
            for existing_drug in existing_drugs:
                existing_drug_lower = existing_drug.lower().strip()
                for drug_name, idx in self.drug_map.items():
                    if existing_drug_lower in drug_name.lower() or drug_name.lower() in existing_drug_lower:
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
                    all_nodes = torch.tensor(list(range(len(self.drug_map))), dtype=torch.long)
                    edge_index_full = torch.tensor([[new_drug_idx, existing_idx], [existing_idx, new_drug_idx]], dtype=torch.long)
                    
                    # Get predictions
                    z = self.gnn_model.encode(all_nodes, edge_index_full)
                    pred = self.gnn_model.decode(z, edge_index)
                    risk_score = torch.sigmoid(pred).item()
                    
                    total_risk += risk_score
                    count += 1
            
            if count > 0:
                avg_risk = (total_risk / count) * 100
                return min(avg_risk, 100.0)  # Cap at 100%
            
            return 0.0
            
        except Exception as e:
            print(f"[ERROR] GNN prediction failed: {e}")
            return 0.0
    
    def _get_rag_interactions(self, new_drug: str, existing_drugs: List[str]) -> List[Dict]:
        """Get documented interactions from RAG system"""
        try:
            interactions = []
            
            for existing_drug in existing_drugs:
                interaction = self.rag_system.search_interaction(new_drug, existing_drug)
                if interaction:
                    interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            print(f"[ERROR] RAG lookup failed: {e}")
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
            smoker = patient_profile.get('is_smoker', 'Not specified')
            alcohol = patient_profile.get('alcohol_consumption', 'Not specified')
            
            context += f"Patient Profile: Name: {name}, Age: {patient_age}, Conditions: {conditions}, "
            context += f"Allergies: {allergies}, Smoker: {smoker}, Alcohol: {alcohol}.\n"
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
        
        # Dosage analysis
        if dosage_info and patient_profile:
            dosage_amount = dosage_info.get('dosage_amount')
            if dosage_amount:
                dosage_analysis = calculate_personalized_dosage(new_drug, dosage_amount, patient_profile)
                context += f"Personalized Dosage Analysis:\n"
                context += f"- Original dose: {dosage_analysis['original_dose']} mg\n"
                context += f"- Suggested dose: {dosage_analysis['suggested_dose']} mg\n"
                context += f"- Adjustment factor: {dosage_analysis['adjustment_factor']}\n"
                if dosage_analysis['adjustment_reasons']:
                    context += f"- Adjustment reasons: {', '.join(dosage_analysis['adjustment_reasons'])}\n"
                context += "\n"
        
        # RAG interactions
        context += "Known Pairwise Interactions from Database (for reference):\n"
        if rag_interactions:
            for interaction in rag_interactions:
                context += f"- {interaction['drug_a']} and {interaction['drug_b']} ({interaction['severity']}): {interaction['interaction']}\n"
        else:
            context += "- No specific pairwise interactions were found in the knowledge base.\n"
        
        return context
    
    def _get_llm_explanation(self, context: str) -> str:
        """Get LLM explanation using existing ask_local_llm function"""
        try:
            explanation = ask_local_llm(context)
            return explanation
        except Exception as e:
            print(f"[ERROR] LLM explanation failed: {e}")
            # Use fallback response generator
            return generate_fallback_response(context)
    
    def _determine_verdict(self, llm_explanation: str, gnn_risk: float, rag_interactions: List[Dict]) -> str:
        """
        Determine verdict using deterministic logic based on GNN risk and RAG interactions.
        This ensures consistency across multiple calls with the same inputs.
        LLM explanation is for human readability only and doesn't affect the verdict.
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
        elif gnn_risk > 30:
            return "CAUTION ADVISED"
        
        return "SAFE TO ADD"
    
    def _validate_dosage(self, drug: str, dosage_info: Optional[Dict]) -> Dict:
        """Validate dosage using dosage validator"""
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
            print(f'[ERROR] Dosage validation failed: {e}')
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
            r'^(hi|hello|hey)[\s!?]*$'
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
    
    response += f"**🤖 AI Risk Analysis**\n"
    response += f"My AI system predicts a {gnn_risk:.1f}% interaction risk"
    if gnn_risk < 20:
        response += " - that's quite low and reassuring!\n\n"
    elif gnn_risk < 50:
        response += " - moderate risk, worth being cautious.\n\n"
    else:
        response += " - this is concerning and requires attention.\n\n"
    
    response += "**💊 Drug Interaction Check**\n"
    if has_interactions:
        response += f"I found documented interactions for {new_drug} in my database. Please review the detailed interaction information above carefully.\n\n"
    else:
        response += f"Good news! I didn't find any documented interactions between {new_drug} and your current medications in my database.\n\n"
    
    response += "**📊 Dosage Safety**\n"
    if has_dosage_warnings:
        response += "⚠️ I have concerns about the dosage you entered. Please review the dosage warnings above.\n\n"
    else:
        response += "The dosage appears to be within normal ranges based on my database.\n\n"
    
    response += "**✅ My Recommendation**\n"
    
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

def ask_local_llm(context):
    prompt = f"""You are Dr. MediBot, a friendly AI health assistant. Analyze this medication safety data and write a clear, simple response.

CRITICAL FORMATTING RULES:
- Write in PLAIN TEXT only
- DO NOT use ** for bold
- DO NOT use ## for headers  
- DO NOT use markdown
- DO NOT use strikethrough
- Just write normal sentences and paragraphs
- Use simple dashes (-) for lists if needed

ANALYSIS DATA:
{context}

Write your response as if you're talking to a friend. Use simple, clear language.

[YOUR ROLE]
You are a caring health assistant who:
- Greets the patient warmly by name
- Explains findings in simple, everyday language  
- Focuses on what matters most to patient safety
- Gives clear, actionable recommendations
- Is reassuring when safe, cautious when risky

[WHAT TO INCLUDE IN YOUR RESPONSE]

1. WARM GREETING
   - Start with "Hi [Name]!"
   - Brief friendly opening

2. AI RISK SCORE (if available in data)
   - State the GNN prediction percentage
   - Explain if it's low, moderate, or high risk

3. DRUG INTERACTIONS
   - Look for "Known Pairwise Interactions from Database"
   - If found: Explain WHAT happens and WHY
   - If not found: Reassure them

4. DOSAGE SAFETY
   - Check "Dosage Information" in the data
   - Compare entered dose vs safe limits
   - Mention if dose is safe or too high

5. SIDE EFFECTS (if in data)
   - List 2-3 common side effects in simple terms
   - Mention any personalized warnings

6. CLEAR RECOMMENDATION
   - Is it safe to add or not?
   - Give specific reasons
   - What should they do?

7. FINAL VERDICT (REQUIRED!)
   End with: "Verdict: SAFE TO ADD" or "Verdict: DO NOT ADD"

[EXAMPLE - WRITE LIKE THIS]

Hi Sarah!

I've analyzed adding Aspirin to your medications. Here's what I found:

My AI system shows a 12% interaction risk - that's quite low.

I checked Aspirin against your current medications (Metformin and Lisinopril) and found no serious interactions. These can be taken together safely.

Your dosage of 100mg once daily is well within safe limits. The maximum safe daily dose is 4000mg.

Aspirin can cause mild stomach upset. Take it with food if this happens.

This looks safe to add! The dosage is appropriate and there are no dangerous interactions.

Verdict: SAFE TO ADD

[REMEMBER]
- Use simple, clear language
- No fancy formatting or markdown
- Be specific about risks and benefits
- Give actionable advice
- Always end with clear verdict
"""
    
    # Try cloud API first (OpenRouter.ai), fallback to localhost
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    
    if api_key:
        # Use OpenRouter.ai cloud API with multiple model fallbacks
        print("[INFO] Using OpenRouter.ai cloud API")
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Medicine Assistant"
        }
        
        # Try multiple models in order of preference
        models_to_try = [
            "nousresearch/hermes-3-llama-3.1-405b:free",  # Most powerful
            "meta-llama/llama-3.2-3b-instruct:free",      # Fast and reliable
            "mistralai/mistral-7b-instruct:free"          # Backup
        ]
        
        for model in models_to_try:
            try:
                print(f"[INFO] Trying model: {model}")
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 2000
                }
                response = requests.post(api_url, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                print(f"[SUCCESS] Model {model} responded successfully")
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                print(f"[WARNING] Model {model} failed: {e}")
                continue  # Try next model
        
        # If all models failed, use fallback
        print("[ERROR] All OpenRouter models failed, using fallback")
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
def get_holistic_context(new_drug, patient, existing_meds, dosage_amount=None, dosage_unit=None, frequency=None):
    patient_age = calculate_age(patient.get('dob'))
    patient['age'] = patient_age
    
    # Safely get patient data with defaults
    name = patient.get('name') or 'Patient'
    conditions = patient.get('conditions') or 'None reported'
    allergies = patient.get('drug_allergies') or 'None reported'
    smoker = patient.get('is_smoker') or 'Not specified'
    alcohol = patient.get('alcohol_consumption') or 'Not specified'
    
    context_str = f"Patient Profile: Name: {name}, Age: {patient_age}, Conditions: {conditions}, Allergies: {allergies}, Smoker: {smoker}, Alcohol: {alcohol}.\n"
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
    
    # Add personalized dosage analysis
    if dosage_amount:
        dosage_analysis = calculate_personalized_dosage(new_drug, dosage_amount, patient)
        context_str += f"Personalized Dosage Analysis:\n"
        context_str += f"- Original dose: {dosage_analysis['original_dose']} mg\n"
        context_str += f"- Suggested dose: {dosage_analysis['suggested_dose']} mg\n"
        context_str += f"- Adjustment factor: {dosage_analysis['adjustment_factor']}\n"
        if dosage_analysis['adjustment_reasons']:
            context_str += f"- Adjustment reasons: {', '.join(dosage_analysis['adjustment_reasons'])}\n"
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
        
        # Create enhanced context including all new features
        holistic_context = get_holistic_context(new_drug, patient, existing_meds, dosage_amount, dosage_unit, frequency)
        
        # Add dosage validation results to context
        if dosage_amount and dosage_unit:
            dosage_info = f"\nDosage Information:\n"
            dosage_info += f"Drug: {new_drug}\n"
            dosage_info += f"Amount: {dosage_amount} {dosage_unit}\n"
            dosage_info += f"Frequency: {frequency or 'Not specified'}\n"
            dosage_info += f"Calculated Daily Dosage: {dosage_validation.get('daily_dosage', 'N/A')} {dosage_validation.get('unit', '')}\n"
            
            if dosage_validation['warnings']:
                dosage_info += f"Dosage Warnings:\n"
                for warning in dosage_validation['warnings']:
                    dosage_info += f"- {warning}\n"
            
            if dosage_validation['max_daily']:
                dosage_info += f"Maximum Safe Daily Dose: {dosage_validation['max_daily']} {dosage_validation['unit']}\n"
            if dosage_validation['max_single']:
                dosage_info += f"Maximum Safe Single Dose: {dosage_validation['max_single']} {dosage_validation['unit']}\n"
            
            holistic_context += dosage_info
        
        # Get GNN risk score
        gnn_risk = 0.0
        if gnn_model and drug_map and existing_meds:
            gnn_risk = get_gnn_prediction(new_drug, existing_meds)
        
        # Get AI summary with error handling
        try:
            ai_summary = ask_local_llm(holistic_context)
            summary_lines = ai_summary.split('\n')
            verdict = summary_lines[-1] if summary_lines else "Verdict: DO NOT ADD"
        except Exception as e:
            print(f"Error getting AI summary: {e}")
            ai_summary = f"I apologize, but I'm having trouble connecting to the AI assistant right now. Please try again in a moment.\n\nVerdict: DO NOT ADD"
            summary_lines = ai_summary.split('\n')
            verdict = summary_lines[-1]
        
        # Determine if medication can be added based on both interaction and dosage safety
        interaction_safe = "SAFE TO ADD" in verdict
        dosage_safe = dosage_validation['is_safe']
        can_add = interaction_safe and dosage_safe
        
        main_summary = "\n".join(summary_lines[:-1]) if len(summary_lines) > 1 else ai_summary
        
        # Format response for React frontend
        response_data = {
            'gnn_risk': round(gnn_risk, 1),
            'verdict': verdict.replace('Verdict: ', ''),
            'ai_response': main_summary,
            'can_add': can_add,
            'dosage_validation': dosage_validation
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
            'dosage_validation': {'is_safe': False, 'warnings': ['Error occurred during validation']}
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
            return jsonify({
                'response': response,
                'intent': 'conversational',
                'timestamp': datetime.now().isoformat()
            })
        
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
        
        return jsonify({
            'response': result.llm_explanation,
            'verdict': result.verdict,
            'gnn_risk': result.gnn_risk,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        })
        
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
        
        # Map verdict to status
        status_map = {
            'SAFE TO ADD': 'SAFE',
            'CAUTION ADVISED': 'CAUTION',
            'DO NOT ADD': 'UNSAFE'
        }
        status = status_map.get(result.verdict, 'UNSAFE')
        
        print(f"[DEBUG] Status: {status}, GNN Risk: {result.gnn_risk}%, Verdict: {result.verdict}")
        
        return jsonify({
            'status': status,
            'response': result.llm_explanation,
            'gnn_risk': result.gnn_risk,
            'drug1': drug1,
            'drug2': drug2,
            'interaction': result.rag_interactions[0] if result.rag_interactions else None,
            'timestamp': result.timestamp
        })
        
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
        
        return jsonify({
            'gnn_risk': result.gnn_risk,
            'verdict': result.verdict,
            'ai_response': result.llm_explanation,
            'can_add': result.can_add,
            'dosage_validation': result.dosage_validation
        })
    
    # Multiple drugs - use InteractionEngine
    # Treat first drug as "new drug" and rest as "existing drugs"
    new_drug = drugs[0]
    existing_drugs = drugs[1:]
    
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None  # Anonymous user
    )
    
    # Map InteractionResult to existing response format for backward compatibility
    return jsonify({
        'gnn_risk': result.gnn_risk,
        'verdict': result.verdict,
        'ai_response': result.llm_explanation,
        'can_add': result.can_add,
        'interactions': result.rag_interactions,
        'dosage_validation': result.dosage_validation
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

