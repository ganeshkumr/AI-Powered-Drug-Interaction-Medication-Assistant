#!/usr/bin/env python3
"""
Tests for anonymous user support
Property-based tests to verify that the system works without authentication
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, interaction_engine, InteractionResult


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

# Feature: backend-improvements, Property 11: Anonymous User Support
@settings(max_examples=3, deadline=None)
@given(
    new_drug=st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril', 
                              'warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol']),
    existing_drugs=st.lists(
        st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril',
                        'warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol']),
        min_size=0,
        max_size=3
    )
)
def test_interaction_engine_works_without_patient_profile(new_drug, existing_drugs):
    """
    Property 11: Anonymous User Support - InteractionEngine
    For any drug combination, the InteractionEngine should work correctly 
    without a patient profile (patient_profile=None).
    
    Validates: Requirements 1.1, 1.2, 2.1
    """
    # Call InteractionEngine without patient profile
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None,  # Anonymous user - no patient profile
        dosage_info=None
    )
    
    # Should return a valid InteractionResult
    assert isinstance(result, InteractionResult), \
        "Should return InteractionResult even without patient profile"
    
    # Should have all required fields
    assert hasattr(result, 'gnn_risk'), "Result should have gnn_risk"
    assert hasattr(result, 'rag_interactions'), "Result should have rag_interactions"
    assert hasattr(result, 'llm_explanation'), "Result should have llm_explanation"
    assert hasattr(result, 'verdict'), "Result should have verdict"
    assert hasattr(result, 'can_add'), "Result should have can_add"
    assert hasattr(result, 'dosage_validation'), "Result should have dosage_validation"
    assert hasattr(result, 'timestamp'), "Result should have timestamp"
    
    # GNN risk should be valid
    assert isinstance(result.gnn_risk, float), "GNN risk should be float"
    assert 0.0 <= result.gnn_risk <= 100.0, \
        f"GNN risk should be 0-100, got {result.gnn_risk}"
    
    # RAG interactions should be a list
    assert isinstance(result.rag_interactions, list), \
        "RAG interactions should be a list"
    
    # LLM explanation should be non-empty
    assert isinstance(result.llm_explanation, str), \
        "LLM explanation should be a string"
    assert len(result.llm_explanation) > 0, \
        "LLM explanation should not be empty"
    
    # Verdict should be valid
    assert result.verdict in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD'], \
        f"Invalid verdict: {result.verdict}"
    
    # can_add should be boolean
    assert isinstance(result.can_add, bool), \
        "can_add should be boolean"
    
    # dosage_validation should be a dict
    assert isinstance(result.dosage_validation, dict), \
        "dosage_validation should be a dict"
    
    # timestamp should be non-empty string
    assert isinstance(result.timestamp, str), \
        "timestamp should be a string"
    assert len(result.timestamp) > 0, \
        "timestamp should not be empty"


@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    drugs=st.lists(
        st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril',
                        'warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol']),
        min_size=1,
        max_size=4,
        unique=True
    )
)
def test_quick_check_works_without_authentication(client, drugs):
    """
    Property 11: Anonymous User Support - Quick Check
    For any drug list, the Quick Check endpoint should work without authentication.
    
    Validates: Requirements 1.1, 2.1
    """
    # Make request to Quick Check without authentication (no session)
    response = client.post('/api/quick-check',
                          json={'drugs': drugs},
                          content_type='application/json')
    
    # Should return 200 OK (not 401 Unauthorized)
    assert response.status_code == 200, \
        f"Quick Check should work without auth, got status {response.status_code}"
    
    # Parse response
    data = response.get_json()
    
    # Should have required fields
    assert 'gnn_risk' in data, "Response should have gnn_risk"
    assert 'verdict' in data, "Response should have verdict"
    assert 'ai_response' in data, "Response should have ai_response"
    assert 'can_add' in data, "Response should have can_add"
    
    # GNN risk should be valid
    assert isinstance(data['gnn_risk'], (int, float)), \
        "GNN risk should be numeric"
    assert 0.0 <= data['gnn_risk'] <= 100.0, \
        f"GNN risk should be 0-100, got {data['gnn_risk']}"
    
    # Verdict should be valid
    assert data['verdict'] in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD'], \
        f"Invalid verdict: {data['verdict']}"
    
    # AI response should be non-empty
    assert isinstance(data['ai_response'], str), \
        "AI response should be a string"
    assert len(data['ai_response']) > 0, \
        "AI response should not be empty"
    
    # can_add should be boolean
    assert isinstance(data['can_add'], bool), \
        "can_add should be boolean"


@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    drug1=st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril',
                          'warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol']),
    drug2=st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril',
                          'warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol'])
)
def test_emergency_check_works_without_authentication(client, drug1, drug2):
    """
    Property 11: Anonymous User Support - Emergency Check
    For any two drugs, the Emergency Check endpoint should work without authentication.
    
    Validates: Requirements 1.2, 2.1
    """
    # Make request to Emergency Check without authentication (no session)
    response = client.post('/emergency-check',
                          json={'drug1': drug1, 'drug2': drug2},
                          content_type='application/json')
    
    # Should return 200 OK (not 401 Unauthorized)
    assert response.status_code == 200, \
        f"Emergency Check should work without auth, got status {response.status_code}"
    
    # Parse response
    data = response.get_json()
    
    # Should have required fields
    assert 'status' in data, "Response should have status"
    assert 'response' in data, "Response should have response"
    assert 'gnn_risk' in data, "Response should have gnn_risk"
    assert 'drug1' in data, "Response should have drug1"
    assert 'drug2' in data, "Response should have drug2"
    assert 'timestamp' in data, "Response should have timestamp"
    
    # Status should be valid
    assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE'], \
        f"Invalid status: {data['status']}"
    
    # Response should be non-empty
    assert isinstance(data['response'], str), \
        "Response should be a string"
    assert len(data['response']) > 0, \
        "Response should not be empty"
    
    # GNN risk should be valid
    assert isinstance(data['gnn_risk'], (int, float)), \
        "GNN risk should be numeric"
    assert 0.0 <= data['gnn_risk'] <= 100.0, \
        f"GNN risk should be 0-100, got {data['gnn_risk']}"
    
    # Drug names should match
    assert data['drug1'] == drug1, \
        f"drug1 should be {drug1}, got {data['drug1']}"
    assert data['drug2'] == drug2, \
        f"drug2 should be {drug2}, got {data['drug2']}"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running Anonymous User Support tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
