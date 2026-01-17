#!/usr/bin/env python3
"""
Tests for /api/quick-check endpoint
Includes both property-based tests and integration tests
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys
import os
import json

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, interaction_engine


# ============================================================================
# TEST SETUP
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

# Feature: backend-improvements, Property 2: Quick Check Uses Full Pipeline
@settings(max_examples=5, deadline=None)
@given(
    drug1=st.sampled_from(['aspirin', 'ibuprofen']),
    drug2=st.sampled_from(['warfarin', 'atorvastatin'])
)
def test_quick_check_uses_full_pipeline(drug1, drug2):
    """
    Property 2: Quick Check Uses Full Pipeline
    For any drug list submitted to Quick Check, the response should include 
    GNN risk prediction, RAG interaction lookup, and LLM-generated explanation 
    (not templated responses).
    
    Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4
    """
    # Create test client inside the test
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Send request to Quick Check endpoint
        response = client.post('/api/quick-check', 
            json={'drugs': [drug1, drug2]},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should have GNN risk prediction
        assert 'gnn_risk' in data
        assert isinstance(data['gnn_risk'], (int, float))
        assert 0 <= data['gnn_risk'] <= 100
        
        # Should have verdict
        assert 'verdict' in data
        assert data['verdict'] in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
        
        # Should have LLM-generated explanation
        assert 'ai_response' in data
        assert len(data['ai_response']) > 0
        
        # Should have can_add field
        assert 'can_add' in data
        assert isinstance(data['can_add'], bool)
        
        # Verify the response came from InteractionEngine
        direct_result = interaction_engine.analyze_interaction(
            new_drug=drug1,
            existing_drugs=[drug2],
            patient_profile=None
        )
        
        # GNN risk should match
        assert data['gnn_risk'] == direct_result.gnn_risk
        
        # Verdict should match
        assert data['verdict'] == direct_result.verdict


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestQuickCheckEndpoint:
    """Integration tests for Quick Check endpoint"""
    
    def test_with_two_drugs_returns_full_analysis(self, client):
        """
        Test with 2 drugs (should return GNN + RAG + LLM)
        Requirements: 1.1, 6.4, 7.3
        """
        response = client.post('/api/quick-check',
            json={'drugs': ['aspirin', 'ibuprofen']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should have all required fields
        assert 'gnn_risk' in data
        assert 'verdict' in data
        assert 'ai_response' in data
        assert 'can_add' in data
        assert 'dosage_validation' in data
        
        # GNN risk should be calculated
        assert isinstance(data['gnn_risk'], (int, float))
        assert data['gnn_risk'] >= 0
        
        # Should have LLM explanation
        assert len(data['ai_response']) > 0
        
        # Verdict should be valid
        assert data['verdict'] in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
    
    def test_with_single_drug_returns_safe_response(self, client):
        """
        Test with single drug (should return safe response)
        Requirements: 1.1, 6.4, 7.3
        """
        response = client.post('/api/quick-check',
            json={'drugs': ['aspirin']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should return safe response for single drug
        assert data['gnn_risk'] == 0.0, "Single drug should have 0 risk"
        assert 'SAFE' in data['verdict'], f"Single drug should be safe, got {data['verdict']}"
        assert data['can_add'] == True, "Single drug should be safe to add"
        
        # Should still have explanation
        assert len(data['ai_response']) > 0
        assert 'aspirin' in data['ai_response'].lower()
    
    def test_with_invalid_drugs_handles_gracefully(self, client):
        """
        Test with invalid drugs (should handle gracefully)
        Requirements: 1.1, 6.4, 7.3
        """
        response = client.post('/api/quick-check',
            json={'drugs': ['INVALID_DRUG_XYZ', 'ANOTHER_INVALID_ABC']},
            content_type='application/json'
        )
        
        # Should not crash
        assert response.status_code == 200
        data = response.get_json()
        
        # Should still return valid response structure
        assert 'gnn_risk' in data
        assert 'verdict' in data
        assert 'ai_response' in data
        assert 'can_add' in data
        
        # Should have a verdict (even if conservative)
        assert data['verdict'] in ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
    
    def test_with_no_drugs_returns_error(self, client):
        """Test with no drugs (should return error)"""
        response = client.post('/api/quick-check',
            json={'drugs': []},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_with_three_drugs(self, client):
        """Test with three drugs"""
        response = client.post('/api/quick-check',
            json={'drugs': ['aspirin', 'ibuprofen', 'paracetamol']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should handle multiple drugs
        assert 'gnn_risk' in data
        assert 'verdict' in data
        assert 'ai_response' in data
    
    def test_response_format_consistency(self, client):
        """Test that response format is consistent"""
        response = client.post('/api/quick-check',
            json={'drugs': ['aspirin', 'warfarin']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Check all required fields exist
        required_fields = ['gnn_risk', 'verdict', 'ai_response', 'can_add', 'dosage_validation']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check types
        assert isinstance(data['gnn_risk'], (int, float))
        assert isinstance(data['verdict'], str)
        assert isinstance(data['ai_response'], str)
        assert isinstance(data['can_add'], bool)
        assert isinstance(data['dosage_validation'], dict)
    
    def test_backward_compatibility_with_frontend(self, client):
        """Test that response format is backward compatible with frontend"""
        response = client.post('/api/quick-check',
            json={'drugs': ['aspirin', 'ibuprofen']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Frontend expects these fields
        assert 'gnn_risk' in data
        assert 'verdict' in data
        assert 'ai_response' in data
        assert 'can_add' in data
        
        # Optional fields that frontend may use
        if 'interactions' in data:
            assert isinstance(data['interactions'], list)
        
        if 'dosage_validation' in data:
            assert isinstance(data['dosage_validation'], dict)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running Quick Check endpoint tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
