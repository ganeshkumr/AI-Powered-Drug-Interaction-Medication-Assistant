#!/usr/bin/env python3
"""
Property-based test for verdict consistency with risk score
Feature: backend-improvements, Property 12: Verdict Consistency with Risk Score
Validates: Requirements 1.3, 7.2
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import interaction_engine, InteractionResult


# ============================================================================
# PROPERTY-BASED TEST
# ============================================================================

# Feature: backend-improvements, Property 12: Verdict Consistency with Risk Score
@settings(max_examples=5, deadline=None)
@given(
    drug1=st.sampled_from(['aspirin', 'ibuprofen', 'metformin']),
    drug2=st.sampled_from(['warfarin', 'atorvastatin', 'omeprazole'])
)
def test_verdict_consistency_with_risk_score(drug1, drug2):
    """
    Property 12: Verdict Consistency with Risk Score
    
    For any interaction analysis, the verdict should be consistent with the GNN risk score:
    - If GNN risk > 70, verdict should be "DO NOT ADD"
    - If GNN risk 30-70, verdict should be "CAUTION ADVISED"
    - If GNN risk < 30, verdict should be "SAFE TO ADD"
    
    This property ensures that the verdict determination logic is deterministic
    and follows the specified risk thresholds.
    
    Validates: Requirements 1.3, 7.2
    """
    # Analyze the interaction
    result = interaction_engine.analyze_interaction(
        new_drug=drug1,
        existing_drugs=[drug2],
        patient_profile=None,
        dosage_info=None
    )
    
    # Verify result is valid
    assert isinstance(result, InteractionResult), "Result should be InteractionResult type"
    
    # Get the risk score and verdict
    gnn_risk = result.gnn_risk
    verdict = result.verdict
    
    # Verify risk is in valid range
    assert 0.0 <= gnn_risk <= 100.0, f"GNN risk {gnn_risk} is out of valid range [0, 100]"
    
    # Verify verdict is one of the valid values
    valid_verdicts = ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
    assert verdict in valid_verdicts, f"Invalid verdict: {verdict}"
    
    # Check RAG interactions for severity (RAG takes precedence over GNN)
    has_major_rag_interaction = False
    has_moderate_rag_interaction = False
    
    for interaction in result.rag_interactions:
        severity = interaction.get('severity', '').lower()
        if severity in ['major', 'severe', 'contraindicated']:
            has_major_rag_interaction = True
            break
        elif severity in ['moderate', 'moderate risk']:
            has_moderate_rag_interaction = True
    
    # Verify verdict consistency based on RAG and GNN risk
    if has_major_rag_interaction:
        # RAG found major interaction - should be DO NOT ADD
        assert verdict == "DO NOT ADD", \
            f"Expected 'DO NOT ADD' for major RAG interaction, got '{verdict}' (GNN risk: {gnn_risk})"
    
    elif has_moderate_rag_interaction:
        # RAG found moderate interaction - should be CAUTION ADVISED
        assert verdict == "CAUTION ADVISED", \
            f"Expected 'CAUTION ADVISED' for moderate RAG interaction, got '{verdict}' (GNN risk: {gnn_risk})"
    
    else:
        # No significant RAG interactions - verdict should be based on GNN risk
        if gnn_risk > 70:
            assert verdict == "DO NOT ADD", \
                f"Expected 'DO NOT ADD' for GNN risk {gnn_risk} > 70, got '{verdict}'"
        
        elif gnn_risk >= 30:
            assert verdict == "CAUTION ADVISED", \
                f"Expected 'CAUTION ADVISED' for GNN risk {gnn_risk} in [30, 70], got '{verdict}'"
        
        else:  # gnn_risk < 30
            assert verdict == "SAFE TO ADD", \
                f"Expected 'SAFE TO ADD' for GNN risk {gnn_risk} < 30, got '{verdict}'"
    
    # Additional check: can_add should be consistent with verdict
    if verdict == "SAFE TO ADD":
        # can_add might be False if dosage validation failed
        # So we only check when dosage is safe
        if result.dosage_validation.get('is_safe', True):
            assert result.can_add == True, \
                f"can_add should be True for 'SAFE TO ADD' verdict with safe dosage"
    elif verdict == "DO NOT ADD":
        # can_add should always be False for DO NOT ADD
        assert result.can_add == False, \
            f"can_add should be False for 'DO NOT ADD' verdict"


# ============================================================================
# UNIT TESTS FOR SPECIFIC RISK THRESHOLDS
# ============================================================================

class TestVerdictConsistencyUnitTests:
    """Unit tests for specific verdict consistency scenarios"""
    
    def test_verdict_matches_risk_boundaries(self):
        """Test verdict consistency at risk boundaries"""
        # Test directly with _determine_verdict method for speed
        
        # Risk = 75 (should be DO NOT ADD)
        verdict_high = interaction_engine._determine_verdict("", 75.0, [])
        assert verdict_high == "DO NOT ADD"
        
        # Risk = 50 (should be CAUTION ADVISED)
        verdict_medium = interaction_engine._determine_verdict("", 50.0, [])
        assert verdict_medium == "CAUTION ADVISED"
        
        # Risk = 25 (should be SAFE TO ADD)
        verdict_low = interaction_engine._determine_verdict("", 25.0, [])
        assert verdict_low == "SAFE TO ADD"
        
        # Risk = 30 (boundary - should be CAUTION ADVISED)
        verdict_boundary_30 = interaction_engine._determine_verdict("", 30.0, [])
        assert verdict_boundary_30 == "CAUTION ADVISED"
        
        # Risk = 71 (just over boundary - should be DO NOT ADD)
        verdict_boundary_71 = interaction_engine._determine_verdict("", 71.0, [])
        assert verdict_boundary_71 == "DO NOT ADD"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running Verdict Consistency tests...")
    print("=" * 60)
    print("Testing Property 12: Verdict Consistency with Risk Score")
    print("Validates: Requirements 1.3, 7.2")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
