"""
Unit tests for response format validation
Tests specific examples and edge cases for response format validation
Requirements: 7.1, 7.2, 7.3, 7.4
"""

import pytest
from response_validation import validate_response_format
from datetime import datetime


class TestQuickCheckResponseFormat:
    """Unit tests for Quick Check response format"""
    
    def test_valid_quick_check_response(self):
        """Test that a valid Quick Check response passes validation"""
        response = {
            'gnn_risk': 25.5,
            'verdict': 'SAFE TO ADD',
            'ai_response': 'This medication appears safe to add.',
            'can_add': True
        }
        assert validate_response_format(response, 'quick_check') is True
    
    def test_quick_check_missing_gnn_risk(self):
        """Test that Quick Check fails without gnn_risk"""
        response = {
            'verdict': 'SAFE TO ADD',
            'ai_response': 'This medication appears safe to add.',
            'can_add': True
        }
        assert validate_response_format(response, 'quick_check') is False
    
    def test_quick_check_missing_verdict(self):
        """Test that Quick Check fails without verdict"""
        response = {
            'gnn_risk': 25.5,
            'ai_response': 'This medication appears safe to add.',
            'can_add': True
        }
        assert validate_response_format(response, 'quick_check') is False
    
    def test_quick_check_missing_ai_response(self):
        """Test that Quick Check fails without ai_response"""
        response = {
            'gnn_risk': 25.5,
            'verdict': 'SAFE TO ADD',
            'can_add': True
        }
        assert validate_response_format(response, 'quick_check') is False
    
    def test_quick_check_missing_can_add(self):
        """Test that Quick Check fails without can_add"""
        response = {
            'gnn_risk': 25.5,
            'verdict': 'SAFE TO ADD',
            'ai_response': 'This medication appears safe to add.'
        }
        assert validate_response_format(response, 'quick_check') is False
    
    def test_quick_check_wrong_type_gnn_risk(self):
        """Test that Quick Check fails with wrong type for gnn_risk"""
        response = {
            'gnn_risk': 'high',  # Should be numeric
            'verdict': 'SAFE TO ADD',
            'ai_response': 'This medication appears safe to add.',
            'can_add': True
        }
        assert validate_response_format(response, 'quick_check') is False
    
    def test_quick_check_wrong_type_can_add(self):
        """Test that Quick Check fails with wrong type for can_add"""
        response = {
            'gnn_risk': 25.5,
            'verdict': 'SAFE TO ADD',
            'ai_response': 'This medication appears safe to add.',
            'can_add': 'yes'  # Should be boolean
        }
        assert validate_response_format(response, 'quick_check') is False


class TestEmergencyCheckResponseFormat:
    """Unit tests for Emergency Check response format"""
    
    def test_valid_emergency_check_response(self):
        """Test that a valid Emergency Check response passes validation"""
        response = {
            'status': 'SAFE',
            'response': 'These medications can be taken together safely.',
            'gnn_risk': 15.0,
            'drug1': 'Aspirin',
            'drug2': 'Ibuprofen'
        }
        assert validate_response_format(response, 'emergency_check') is True
    
    def test_emergency_check_caution_status(self):
        """Test Emergency Check with CAUTION status"""
        response = {
            'status': 'CAUTION',
            'response': 'Use caution when combining these medications.',
            'gnn_risk': 45.0,
            'drug1': 'Aspirin',
            'drug2': 'Warfarin'
        }
        assert validate_response_format(response, 'emergency_check') is True
    
    def test_emergency_check_unsafe_status(self):
        """Test Emergency Check with UNSAFE status"""
        response = {
            'status': 'UNSAFE',
            'response': 'Do not combine these medications.',
            'gnn_risk': 85.0,
            'drug1': 'Drug A',
            'drug2': 'Drug B'
        }
        assert validate_response_format(response, 'emergency_check') is True
    
    def test_emergency_check_invalid_status(self):
        """Test that Emergency Check fails with invalid status"""
        response = {
            'status': 'MAYBE',  # Invalid status
            'response': 'These medications might interact.',
            'gnn_risk': 50.0,
            'drug1': 'Aspirin',
            'drug2': 'Ibuprofen'
        }
        assert validate_response_format(response, 'emergency_check') is False
    
    def test_emergency_check_missing_drug1(self):
        """Test that Emergency Check fails without drug1"""
        response = {
            'status': 'SAFE',
            'response': 'These medications can be taken together safely.',
            'gnn_risk': 15.0,
            'drug2': 'Ibuprofen'
        }
        assert validate_response_format(response, 'emergency_check') is False
    
    def test_emergency_check_missing_drug2(self):
        """Test that Emergency Check fails without drug2"""
        response = {
            'status': 'SAFE',
            'response': 'These medications can be taken together safely.',
            'gnn_risk': 15.0,
            'drug1': 'Aspirin'
        }
        assert validate_response_format(response, 'emergency_check') is False


class TestChatbotMedicalResponseFormat:
    """Unit tests for Chatbot medical response format"""
    
    def test_valid_chatbot_medical_response(self):
        """Test that a valid Chatbot medical response passes validation"""
        response = {
            'response': 'Based on my analysis, this medication is safe to add.',
            'verdict': 'SAFE TO ADD',
            'gnn_risk': 20.0,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is True
    
    def test_chatbot_medical_missing_response(self):
        """Test that Chatbot medical fails without response"""
        response = {
            'verdict': 'SAFE TO ADD',
            'gnn_risk': 20.0,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is False
    
    def test_chatbot_medical_missing_verdict(self):
        """Test that Chatbot medical fails without verdict"""
        response = {
            'response': 'Based on my analysis, this medication is safe to add.',
            'gnn_risk': 20.0,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is False
    
    def test_chatbot_medical_missing_gnn_risk(self):
        """Test that Chatbot medical fails without gnn_risk"""
        response = {
            'response': 'Based on my analysis, this medication is safe to add.',
            'verdict': 'SAFE TO ADD',
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is False
    
    def test_chatbot_medical_missing_intent(self):
        """Test that Chatbot medical fails without intent"""
        response = {
            'response': 'Based on my analysis, this medication is safe to add.',
            'verdict': 'SAFE TO ADD',
            'gnn_risk': 20.0,
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is False
    
    def test_chatbot_medical_wrong_intent(self):
        """Test that Chatbot medical fails with wrong intent"""
        response = {
            'response': 'Based on my analysis, this medication is safe to add.',
            'verdict': 'SAFE TO ADD',
            'gnn_risk': 20.0,
            'intent': 'conversational',  # Should be 'medical'
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_medical') is False


class TestChatbotConversationalResponseFormat:
    """Unit tests for Chatbot conversational response format"""
    
    def test_valid_chatbot_conversational_response(self):
        """Test that a valid Chatbot conversational response passes validation"""
        response = {
            'response': 'Hi! How can I help you today?',
            'intent': 'conversational',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_conversational') is True
    
    def test_chatbot_conversational_missing_response(self):
        """Test that Chatbot conversational fails without response"""
        response = {
            'intent': 'conversational',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_conversational') is False
    
    def test_chatbot_conversational_missing_intent(self):
        """Test that Chatbot conversational fails without intent"""
        response = {
            'response': 'Hi! How can I help you today?',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_conversational') is False
    
    def test_chatbot_conversational_wrong_intent(self):
        """Test that Chatbot conversational fails with wrong intent"""
        response = {
            'response': 'Hi! How can I help you today?',
            'intent': 'medical',  # Should be 'conversational'
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_conversational') is False
    
    def test_chatbot_conversational_wrong_type_response(self):
        """Test that Chatbot conversational fails with wrong type for response"""
        response = {
            'response': 123,  # Should be string
            'intent': 'conversational',
            'timestamp': datetime.now().isoformat()
        }
        assert validate_response_format(response, 'chatbot_conversational') is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
