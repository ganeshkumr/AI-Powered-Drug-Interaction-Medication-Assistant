#!/usr/bin/env python3
"""
Tests for /emergency-check endpoint
Includes both property-based tests and integration tests
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime
import sys
import os
import json

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, interaction_engine


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

# Feature: backend-improvements, Property 3: Emergency Check Uses Full Pipeline
@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    drug1=st.sampled_from(['aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'lisinopril']),
    drug2=st.sampled_from(['warfarin', 'atorvastatin', 'omeprazole', 'amlodipine', 'metoprolol'])
)
def test_emergency_check_uses_full_pipeline(client, drug1, drug2):
    """
    Property 3: Emergency Check Uses Full Pipeline
    For any two drugs submitted to Emergency Check, the response should include 
    GNN risk prediction, RAG interaction lookup, and LLM-generated explanation.
    
    Validates: Requirements 1.2, 2.1, 2.2, 2.3, 2.4
    """
    # Make request to emergency check endpoint
    response = client.post('/emergency-check', 
                          json={'drug1': drug1, 'drug2': drug2},
                          content_type='application/json')
    
    # Should return 200 OK
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Parse response
    data = json.loads(response.data)
    
    # Should have required fields
    assert 'status' in data, "Response should have 'status' field"
    assert 'response' in data, "Response should have 'response' field (LLM explanation)"
    assert 'gnn_risk' in data, "Response should have 'gnn_risk' field (GNN prediction)"
    
    # GNN risk should be present and valid
    assert isinstance(data['gnn_risk'], (int, float)), "GNN risk should be numeric"
    assert 0.0 <= data['gnn_risk'] <= 100.0, f"GNN risk should be 0-100, got {data['gnn_risk']}"
    
    # Response should be non-empty (LLM explanation)
    assert len(data['response']) > 0, "Response should have LLM-generated explanation"
    
    # Status should be valid
    assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE'], \
        f"Invalid status: {data['status']}"
    
    # Should have timestamp
    assert 'timestamp' in data, "Response should have timestamp"
    
    # Timestamp should be ISO format
    try:
        datetime.fromisoformat(data['timestamp'])
    except ValueError:
        pytest.fail("Timestamp is not in ISO format")
    
    # Should include drug names
    assert 'drug1' in data
    assert 'drug2' in data
    assert data['drug1'] == drug1
    assert data['drug2'] == drug2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEmergencyCheckEndpoint:
    """Integration tests for Emergency Check endpoint"""
    
    def test_with_two_drugs_returns_full_analysis(self, client):
        """
        Test with 2 drugs (should return GNN + RAG + LLM)
        Requirements: 1.2, 6.4, 7.3
        """
        response = client.post('/emergency-check',
                              json={'drug1': 'aspirin', 'drug2': 'warfarin'},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have all required fields
        assert 'status' in data
        assert 'response' in data
        assert 'gnn_risk' in data
        assert 'drug1' in data
        assert 'drug2' in data
        assert 'timestamp' in data
        
        # GNN risk should be present
        assert isinstance(data['gnn_risk'], (int, float))
        assert data['gnn_risk'] >= 0.0
        
        # Should have LLM explanation
        assert len(data['response']) > 0
        
        # Status should be valid
        assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE']
    
    def test_with_invalid_drugs_handles_gracefully(self, client):
        """
        Test with invalid drugs (should handle gracefully)
        Requirements: 1.2, 6.4, 7.3
        """
        response = client.post('/emergency-check',
                              json={'drug1': 'INVALID_DRUG_XYZ', 'drug2': 'ANOTHER_INVALID_ABC'},
                              content_type='application/json')
        
        # Should not crash
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have required fields
        assert 'status' in data
        assert 'response' in data
        assert 'gnn_risk' in data
        
        # Should have a valid status
        assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE']
        
        # Should have an explanation
        assert len(data['response']) > 0
    
    def test_status_mapping_verdict_to_status(self, client):
        """
        Test status mapping (verdict to status conversion)
        Requirements: 1.2, 6.4, 7.3
        """
        # Test with known drug pair
        response = client.post('/emergency-check',
                              json={'drug1': 'aspirin', 'drug2': 'ibuprofen'},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Status should be one of the valid values
        assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE']
        
        # Status should be consistent with GNN risk
        # High risk should map to UNSAFE or CAUTION
        if data['gnn_risk'] > 70:
            assert data['status'] in ['UNSAFE', 'CAUTION'], \
                f"High risk ({data['gnn_risk']}) should map to UNSAFE or CAUTION, got {data['status']}"
    
    def test_missing_drug1_returns_error(self, client):
        """Test with missing drug1 parameter"""
        response = client.post('/emergency-check',
                              json={'drug2': 'aspirin'},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'UNSAFE'
    
    def test_missing_drug2_returns_error(self, client):
        """Test with missing drug2 parameter"""
        response = client.post('/emergency-check',
                              json={'drug1': 'aspirin'},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'UNSAFE'
    
    def test_empty_drug_names_returns_error(self, client):
        """Test with empty drug names"""
        response = client.post('/emergency-check',
                              json={'drug1': '', 'drug2': ''},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'UNSAFE'
    
    def test_no_data_returns_error(self, client):
        """Test with no JSON data"""
        response = client.post('/emergency-check',
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'UNSAFE'
    
    def test_response_format_matches_frontend_expectations(self, client):
        """Test that response format is compatible with frontend"""
        response = client.post('/emergency-check',
                              json={'drug1': 'aspirin', 'drug2': 'ibuprofen'},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Frontend expects these fields
        required_fields = ['status', 'response', 'gnn_risk', 'drug1', 'drug2', 'timestamp']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Status should be one of the expected values
        assert data['status'] in ['SAFE', 'CAUTION', 'UNSAFE']
        
        # Response should be a string
        assert isinstance(data['response'], str)
        
        # GNN risk should be numeric
        assert isinstance(data['gnn_risk'], (int, float))
        
        # Drug names should be strings
        assert isinstance(data['drug1'], str)
        assert isinstance(data['drug2'], str)
        
        # Timestamp should be string
        assert isinstance(data['timestamp'], str)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running Emergency Check endpoint tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
