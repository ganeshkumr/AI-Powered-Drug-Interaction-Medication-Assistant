#!/usr/bin/env python3
"""
Tests for InteractionEngine class
Includes both property-based tests and unit tests
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    interaction_engine, 
    gnn_model, 
    drug_map, 
    rag_system,
    InteractionResult
)


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

# Feature: backend-improvements, Property 1: Interaction Engine Consistency
@settings(max_examples=5, deadline=None)
@given(
    drug1=st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril']),
    drug2=st.sampled_from(['warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol'])
)
def test_interaction_engine_consistency(drug1, drug2):
    """
    Property 1: Interaction Engine Consistency
    For any drug combination, calling the Interaction Engine multiple times 
    with the same inputs should produce the same GNN risk score, RAG interactions, 
    and verdict (LLM explanation may vary slightly due to temperature, but verdict 
    should be consistent).
    
    Validates: Requirements 1.1, 1.2, 2.5
    """
    # Call the engine twice with the same inputs
    result1 = interaction_engine.analyze_interaction(
        new_drug=drug1,
        existing_drugs=[drug2],
        patient_profile=None,
        dosage_info=None
    )
    
    result2 = interaction_engine.analyze_interaction(
        new_drug=drug1,
        existing_drugs=[drug2],
        patient_profile=None,
        dosage_info=None
    )
    
    # GNN risk should be identical
    assert result1.gnn_risk == result2.gnn_risk, \
        f"GNN risk inconsistent: {result1.gnn_risk} vs {result2.gnn_risk}"
    
    # RAG interactions should be identical
    assert len(result1.rag_interactions) == len(result2.rag_interactions), \
        f"RAG interactions count inconsistent: {len(result1.rag_interactions)} vs {len(result2.rag_interactions)}"
    
    # Verdict should be consistent (deterministic based on GNN + RAG)
    # Note: We're now using deterministic verdict logic, so this should always pass
    assert result1.verdict == result2.verdict, \
        f"Verdict inconsistent: {result1.verdict} vs {result2.verdict} (GNN risk: {result1.gnn_risk})"
    
    # Result should be InteractionResult type
    assert isinstance(result1, InteractionResult)
    assert isinstance(result2, InteractionResult)
    
    # Timestamp should be ISO format
    try:
        datetime.fromisoformat(result1.timestamp)
        datetime.fromisoformat(result2.timestamp)
    except ValueError:
        pytest.fail("Timestamp is not in ISO format")


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestInteractionEngineEdgeCases:
    """Unit tests for InteractionEngine edge cases"""
    
    def test_single_drug_returns_low_risk(self):
        """
        Test with single drug (should return low risk)
        Requirements: 1.1, 1.2, 6.4
        """
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=[],  # No existing drugs
            patient_profile=None,
            dosage_info=None
        )
        
        # Should return low/zero risk
        assert result.gnn_risk == 0.0, f"Expected 0.0 risk for single drug, got {result.gnn_risk}"
        
        # Should have no RAG interactions
        assert len(result.rag_interactions) == 0, "Should have no interactions for single drug"
        
        # Should have an explanation
        assert len(result.llm_explanation) > 0, "Should have an explanation"
        assert 'aspirin' in result.llm_explanation.lower(), "Explanation should mention the drug"
        
        # Should be InteractionResult type
        assert isinstance(result, InteractionResult)
    
    def test_invalid_drug_names_handled_gracefully(self):
        """
        Test with invalid drug names (should handle gracefully)
        Requirements: 1.1, 1.2, 6.4
        """
        result = interaction_engine.analyze_interaction(
            new_drug='INVALID_DRUG_XYZ_123',
            existing_drugs=['ANOTHER_INVALID_DRUG_ABC_456'],
            patient_profile=None,
            dosage_info=None
        )
        
        # Should not crash and return a result
        assert isinstance(result, InteractionResult)
        
        # Should have an explanation (even if it's a fallback)
        assert len(result.llm_explanation) > 0, "Should have an explanation even for invalid drugs"
        
        # Should have a verdict
        assert result.verdict in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD'], \
            f"Invalid verdict: {result.verdict}"
        
        # GNN risk should be 0 or low (since drugs not in map)
        assert result.gnn_risk >= 0.0, "Risk should be non-negative"
        assert result.gnn_risk <= 100.0, "Risk should not exceed 100%"
    
    def test_missing_patient_profile_works(self):
        """
        Test with missing patient profile (should work)
        Requirements: 1.1, 1.2, 6.4
        """
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=['ibuprofen'],
            patient_profile=None,  # No patient profile
            dosage_info=None
        )
        
        # Should work without patient profile
        assert isinstance(result, InteractionResult)
        
        # Should have all required fields
        assert hasattr(result, 'gnn_risk')
        assert hasattr(result, 'rag_interactions')
        assert hasattr(result, 'llm_explanation')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'can_add')
        assert hasattr(result, 'dosage_validation')
        assert hasattr(result, 'timestamp')
        
        # Should have valid values
        assert result.gnn_risk >= 0.0
        assert isinstance(result.rag_interactions, list)
        assert len(result.llm_explanation) > 0
        assert result.verdict in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
        assert isinstance(result.can_add, bool)
        assert isinstance(result.dosage_validation, dict)
    
    def test_multiple_existing_drugs(self):
        """Test with multiple existing drugs"""
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=['ibuprofen', 'paracetamol', 'metformin'],
            patient_profile=None,
            dosage_info=None
        )
        
        # Should handle multiple drugs
        assert isinstance(result, InteractionResult)
        assert result.gnn_risk >= 0.0
        assert result.gnn_risk <= 100.0
    
    def test_with_dosage_info(self):
        """Test with dosage information provided"""
        dosage_info = {
            'dosage_amount': '500',
            'dosage_unit': 'mg',
            'frequency': 'twice daily'
        }
        
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=['ibuprofen'],
            patient_profile=None,
            dosage_info=dosage_info
        )
        
        # Should include dosage validation
        assert isinstance(result.dosage_validation, dict)
        assert 'is_safe' in result.dosage_validation
        assert 'warnings' in result.dosage_validation
    
    def test_with_patient_profile(self):
        """Test with patient profile provided"""
        patient_profile = {
            'name': 'Test Patient',
            'dob': '1980-01-01',
            'conditions': 'hypertension',
            'drug_allergies': 'None',
            'is_smoker': 'No',
            'alcohol_consumption': 'None'
        }
        
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=['ibuprofen'],
            patient_profile=patient_profile,
            dosage_info=None
        )
        
        # Should work with patient profile
        assert isinstance(result, InteractionResult)
        # Explanation might include patient-specific information
        assert len(result.llm_explanation) > 0
    
    def test_result_structure(self):
        """Test that result has correct structure"""
        result = interaction_engine.analyze_interaction(
            new_drug='aspirin',
            existing_drugs=['ibuprofen'],
            patient_profile=None,
            dosage_info=None
        )
        
        # Check all required fields exist
        assert hasattr(result, 'gnn_risk')
        assert hasattr(result, 'rag_interactions')
        assert hasattr(result, 'llm_explanation')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'can_add')
        assert hasattr(result, 'dosage_validation')
        assert hasattr(result, 'timestamp')
        
        # Check types
        assert isinstance(result.gnn_risk, float)
        assert isinstance(result.rag_interactions, list)
        assert isinstance(result.llm_explanation, str)
        assert isinstance(result.verdict, str)
        assert isinstance(result.can_add, bool)
        assert isinstance(result.dosage_validation, dict)
        assert isinstance(result.timestamp, str)
        
        # Check value ranges
        assert 0.0 <= result.gnn_risk <= 100.0
        assert result.verdict in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running InteractionEngine tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
