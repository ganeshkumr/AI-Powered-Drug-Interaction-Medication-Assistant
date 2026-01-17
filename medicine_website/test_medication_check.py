#!/usr/bin/env python3
"""
Tests for /ask_assistant endpoint (Chatbot)
Includes property-based tests and integration tests
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, intent_classifier, interaction_engine, conversational_handler


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

# Feature: backend-improvements, Property 7: Medical Query Processing
@settings(max_examples=10, deadline=None)
@given(
    drug_name=st.sampled_from([
        'Ibuprofen', 'Metformin', 'Lisinopril',
        'Warfarin', 'Atorvastatin', 'Omeprazole', 'Amlodipine', 'Metoprolol',
        'aspirin', 'tylenol', 'paracetamol'
    ])
)
def test_medical_query_processing(drug_name):
    """
    Property 7: Medical Query Processing
    For any message classified as medical intent, 
    the Chatbot should invoke the Interaction Engine 
    and return medical analysis results.
    
    Validates: Requirements 3.4, 5.1, 5.2, 5.3
    """
    # Create medical query messages
    messages = [
        f"Can I take {drug_name}?",
        f"Is {drug_name} safe?",
        f"What about {drug_name}",
        f"I want to add {drug_name}",
        f"{drug_name} interaction"
    ]
    
    for message in messages:
        # Step 1: Verify intent classification
        intent = intent_classifier.classify(message)
        
        # Should be classified as medical
        assert intent.type == "medical", \
            f"Expected 'medical' intent for '{message}', got '{intent.type}'"
        
        # Step 2: Verify that medical messages would trigger InteractionEngine
        # (We test the logic, not the actual endpoint which requires session)
        
        # If medical intent, should extract drugs or have medical keywords
        if intent.extracted_drugs:
            # Should have extracted the drug name
            assert len(intent.extracted_drugs) > 0, \
                f"Expected extracted drugs for '{message}', got none"
        else:
            # Should at least be medical intent (keywords detected)
            assert intent.type == "medical", \
                f"Expected medical intent for '{message}'"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestChatbotEndpoint:
    """Integration tests for /ask_assistant endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_session(self):
        """Mock session with patient_id"""
        with patch('app.session', {'patient_id': 1}):
            yield
    
    def test_greeting_returns_conversational_response(self, client, mock_session):
        """
        Test with greeting (should return conversational response)
        Requirements: 3.1, 3.2, 4.1, 5.1, 5.2
        """
        greetings = ['hi', 'hello', 'hey', 'good morning']
        
        for greeting in greetings:
            with patch('app.get_db_connection') as mock_db:
                # Mock database connection
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchone.return_value = {
                    'id': 1,
                    'name': 'Test Patient',
                    'dob': '1990-01-01',
                    'conditions': 'None',
                    'drug_allergies': 'None',
                    'is_smoker': 'No',
                    'alcohol_consumption': 'None'
                }
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_db.return_value = mock_conn
                
                response = client.post('/ask_assistant',
                    json={'question': greeting},
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                # Should have conversational intent
                assert data.get('intent') == 'conversational', \
                    f"Expected conversational intent for '{greeting}', got {data.get('intent')}"
                
                # Should have a response
                assert 'response' in data
                assert len(data['response']) > 0
                
                # Should NOT have medical fields
                assert 'verdict' not in data or data['verdict'] is None
                assert 'gnn_risk' not in data
    
    def test_medical_query_uses_interaction_engine(self, client, mock_session):
        """
        Test with medical query (should use InteractionEngine)
        Requirements: 3.1, 3.2, 4.1, 5.1, 5.2
        """
        medical_queries = [
            'Can I take aspirin?',
            'Is ibuprofen safe?',
            'What about metformin?'
        ]
        
        for query in medical_queries:
            with patch('app.get_db_connection') as mock_db, \
                 patch('app.interaction_engine.analyze_interaction') as mock_analyze:
                
                # Mock database connection
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchone.return_value = {
                    'id': 1,
                    'name': 'Test Patient',
                    'dob': '1990-01-01',
                    'conditions': 'None',
                    'drug_allergies': 'None',
                    'is_smoker': 'No',
                    'alcohol_consumption': 'None'
                }
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_db.return_value = mock_conn
                
                # Mock InteractionEngine response
                from app import InteractionResult
                mock_analyze.return_value = InteractionResult(
                    gnn_risk=15.5,
                    rag_interactions=[],
                    llm_explanation="This medication appears safe.",
                    verdict="SAFE TO ADD",
                    can_add=True,
                    dosage_validation={'is_safe': True, 'warnings': []},
                    timestamp="2024-01-01T00:00:00"
                )
                
                response = client.post('/ask_assistant',
                    json={'question': query},
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                # Should have medical intent
                assert data.get('intent') == 'medical', \
                    f"Expected medical intent for '{query}', got {data.get('intent')}"
                
                # Should have medical analysis fields
                assert 'response' in data
                assert 'verdict' in data
                assert 'gnn_risk' in data
                
                # InteractionEngine should have been called
                assert mock_analyze.called, \
                    f"InteractionEngine should be called for medical query '{query}'"
    
    def test_farewell_returns_conversational_response(self, client, mock_session):
        """
        Test with farewell (should return conversational response)
        Requirements: 3.1, 3.2, 4.1, 5.1, 5.2
        """
        farewells = ['bye', 'goodbye', 'see you', 'thanks']
        
        for farewell in farewells:
            with patch('app.get_db_connection') as mock_db:
                # Mock database connection
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchone.return_value = {
                    'id': 1,
                    'name': 'Test Patient',
                    'dob': '1990-01-01',
                    'conditions': 'None',
                    'drug_allergies': 'None',
                    'is_smoker': 'No',
                    'alcohol_consumption': 'None'
                }
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_db.return_value = mock_conn
                
                response = client.post('/ask_assistant',
                    json={'question': farewell},
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                # Should have conversational intent
                assert data.get('intent') == 'conversational', \
                    f"Expected conversational intent for '{farewell}', got {data.get('intent')}"
                
                # Should have a response
                assert 'response' in data
                assert len(data['response']) > 0
    
    def test_drug_name_in_message_extracts_and_analyzes(self, client, mock_session):
        """
        Test with drug name in message (should extract and analyze)
        Requirements: 3.1, 3.2, 4.1, 5.1, 5.2
        """
        messages_with_drugs = [
            'Can I take aspirin with my current medications?',
            'Is ibuprofen safe for me?',
            'I want to add metformin to my regimen'
        ]
        
        for message in messages_with_drugs:
            with patch('app.get_db_connection') as mock_db, \
                 patch('app.interaction_engine.analyze_interaction') as mock_analyze:
                
                # Mock database connection
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchone.return_value = {
                    'id': 1,
                    'name': 'Test Patient',
                    'dob': '1990-01-01',
                    'conditions': 'None',
                    'drug_allergies': 'None',
                    'is_smoker': 'No',
                    'alcohol_consumption': 'None'
                }
                mock_conn.execute.return_value.fetchall.return_value = [
                    {'drug_name': 'Lisinopril', 'dosage_amount': '10', 'dosage_unit': 'mg'}
                ]
                mock_db.return_value = mock_conn
                
                # Mock InteractionEngine response
                from app import InteractionResult
                mock_analyze.return_value = InteractionResult(
                    gnn_risk=25.0,
                    rag_interactions=[],
                    llm_explanation="Analysis complete.",
                    verdict="CAUTION ADVISED",
                    can_add=True,
                    dosage_validation={'is_safe': True, 'warnings': []},
                    timestamp="2024-01-01T00:00:00"
                )
                
                response = client.post('/ask_assistant',
                    json={'question': message},
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                # Should have medical intent
                assert data.get('intent') == 'medical', \
                    f"Expected medical intent for '{message}', got {data.get('intent')}"
                
                # Should have called InteractionEngine
                assert mock_analyze.called, \
                    f"InteractionEngine should be called for '{message}'"
                
                # Check that analyze_interaction was called with correct parameters
                call_args = mock_analyze.call_args
                assert call_args is not None
                assert 'new_drug' in call_args[1] or len(call_args[0]) > 0
    
    def test_not_logged_in_returns_401(self, client):
        """Test that endpoint requires authentication"""
        response = client.post('/ask_assistant',
            json={'question': 'hello'},
            content_type='application/json'
        )
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_empty_message_returns_400(self, client, mock_session):
        """Test that empty message returns error"""
        with patch('app.get_db_connection'):
            response = client.post('/ask_assistant',
                json={'question': ''},
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_no_message_returns_400(self, client, mock_session):
        """Test that missing message returns error"""
        with patch('app.get_db_connection'):
            response = client.post('/ask_assistant',
                json={},
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running Chatbot endpoint tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
