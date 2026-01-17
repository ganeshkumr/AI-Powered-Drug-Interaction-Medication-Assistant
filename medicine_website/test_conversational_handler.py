#!/usr/bin/env python3
"""
Tests for ConversationalHandler class
Includes both property-based tests and unit tests
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import conversational_handler, intent_classifier, Intent


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

# Feature: backend-improvements, Property 6: Conversational Response Appropriateness
@settings(max_examples=10, deadline=None)
@given(
    message=st.sampled_from([
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'Hi', 'Hello', 'Hey', 'HELLO', 'HI',
        'bye', 'goodbye', 'see you', 'thanks', 'thank you',
        'how are you', 'how are you?', 'How are you?',
        'who are you', 'who are you?', 'Who are you?',
        'help', 'Help', 'HELP', 'help!', 'help?',
        'what can you do', 'what can you do?', 'What can you do?'
    ])
)
def test_conversational_response_appropriateness(message):
    """
    Property 6: Conversational Response Appropriateness
    For any message classified as conversational intent, 
    the Chatbot should NOT invoke the Interaction Engine 
    and should return a conversational response.
    
    Validates: Requirements 3.3, 4.1, 4.2, 4.3, 4.4
    """
    # First, verify the message is classified as conversational
    intent = intent_classifier.classify(message)
    
    # Should be conversational type
    assert intent.type == "conversational", \
        f"Expected 'conversational' for '{message}', got '{intent.type}'"
    
    # Get response from conversational handler
    response = conversational_handler.handle(message, intent)
    
    # Response should be a non-empty string
    assert isinstance(response, str), \
        f"Expected string response, got {type(response)}"
    assert len(response) > 0, \
        f"Expected non-empty response for '{message}'"
    
    # Response should be friendly and appropriate
    # Check that response doesn't contain error messages or medical analysis
    assert "error" not in response.lower(), \
        f"Response should not contain errors for conversational message '{message}'"
    assert "gnn" not in response.lower(), \
        f"Response should not contain GNN analysis for conversational message '{message}'"
    # Note: "interaction" can appear in help responses when describing capabilities
    # but should not appear as actual interaction analysis (e.g., "GNN Predicted Risk", "Verdict:")
    assert "gnn predicted risk" not in response.lower(), \
        f"Response should not contain GNN risk analysis for conversational message '{message}'"
    assert "verdict:" not in response.lower(), \
        f"Response should not contain verdict for conversational message '{message}'"
    
    # Response should be welcoming/helpful
    # Check for positive indicators (emojis, friendly words)
    friendly_indicators = ['👋', '😊', '🤖', 'help', 'assist', 'hi', 'hello', 'welcome', 'glad', 'happy']
    has_friendly_indicator = any(indicator in response.lower() for indicator in friendly_indicators)
    assert has_friendly_indicator, \
        f"Response should be friendly for conversational message '{message}', got: {response}"


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestConversationalHandlerResponses:
    """Unit tests for ConversationalHandler responses"""
    
    def test_greeting_responses_are_welcoming(self):
        """
        Test greeting responses (should be welcoming)
        Requirements: 4.1
        """
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        
        for greeting in greetings:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(greeting, intent)
            
            # Should be welcoming
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should contain welcoming elements
            welcoming_words = ['hi', 'hello', 'hey', 'welcome', 'assist', 'help']
            has_welcoming = any(word in response.lower() for word in welcoming_words)
            assert has_welcoming, \
                f"Greeting response should be welcoming for '{greeting}', got: {response}"
            
            # Should offer assistance
            assistance_words = ['help', 'assist', 'can i', 'what would you', 'how can i']
            has_assistance = any(word in response.lower() for word in assistance_words)
            assert has_assistance, \
                f"Greeting response should offer assistance for '{greeting}', got: {response}"
    
    def test_farewell_responses_are_polite(self):
        """
        Test farewell responses (should be polite)
        Requirements: 4.4
        """
        farewells = ['bye', 'goodbye', 'see you', 'thanks', 'thank you']
        
        for farewell in farewells:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(farewell, intent)
            
            # Should be polite
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should contain polite elements
            polite_words = ['take care', 'goodbye', 'bye', 'see you', 'welcome', 'thank', 'glad', 'happy', 'pleasure']
            has_polite = any(word in response.lower() for word in polite_words)
            assert has_polite, \
                f"Farewell response should be polite for '{farewell}', got: {response}"
    
    def test_help_responses_list_capabilities(self):
        """
        Test help responses (should list capabilities)
        Requirements: 4.3
        """
        help_messages = ['help', 'what can you do', 'how can you help']
        
        for message in help_messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should list capabilities
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should mention key capabilities
            capabilities = ['drug', 'interaction', 'medication', 'safe']
            has_capabilities = any(cap in response.lower() for cap in capabilities)
            assert has_capabilities, \
                f"Help response should list capabilities for '{message}', got: {response}"
            
            # Should provide examples
            example_indicators = ['ask', 'try', 'like:', 'example', 'can i']
            has_examples = any(indicator in response.lower() for indicator in example_indicators)
            assert has_examples, \
                f"Help response should provide examples for '{message}', got: {response}"
    
    def test_unclear_message_responses_ask_for_clarification(self):
        """
        Test unclear message responses (should ask for clarification)
        Requirements: 4.5
        """
        unclear_messages = ['what', 'huh', 'i dont know', 'maybe', 'interesting']
        
        for message in unclear_messages:
            intent = Intent(type="conversational", confidence=0.6, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should ask for clarification
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should indicate confusion or ask for clarification
            clarification_words = ['not sure', 'clarify', 'understand', 'asking', 'specific', 'try asking']
            has_clarification = any(word in response.lower() for word in clarification_words)
            assert has_clarification, \
                f"Unclear response should ask for clarification for '{message}', got: {response}"
    
    def test_how_are_you_responses(self):
        """Test responses to 'how are you' questions"""
        messages = ['how are you', 'how are you?', 'How are you?']
        
        for message in messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should respond conversationally
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should redirect to medical assistance
            redirect_words = ['help', 'assist', 'medication', 'question']
            has_redirect = any(word in response.lower() for word in redirect_words)
            assert has_redirect, \
                f"'How are you' response should redirect to assistance for '{message}', got: {response}"
    
    def test_who_are_you_responses(self):
        """Test responses to 'who are you' questions"""
        messages = ['who are you', 'who are you?', 'what are you']
        
        for message in messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should introduce itself
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should mention being an AI assistant
            identity_words = ['ai', 'assistant', 'medibot', 'bot', 'help']
            has_identity = any(word in response.lower() for word in identity_words)
            assert has_identity, \
                f"'Who are you' response should introduce itself for '{message}', got: {response}"
            
            # Should mention capabilities
            capability_words = ['drug', 'interaction', 'medication', 'safety']
            has_capability = any(word in response.lower() for word in capability_words)
            assert has_capability, \
                f"'Who are you' response should mention capabilities for '{message}', got: {response}"
    
    def test_response_randomization(self):
        """Test that responses are randomized to avoid repetition"""
        message = 'hi'
        intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
        
        # Get multiple responses
        responses = [conversational_handler.handle(message, intent) for _ in range(10)]
        
        # Should have some variation (not all identical)
        unique_responses = set(responses)
        assert len(unique_responses) > 1, \
            "Responses should be randomized to avoid repetition"
    
    def test_response_structure(self):
        """Test that all responses have proper structure"""
        test_messages = [
            'hi', 'bye', 'help', 'how are you', 'who are you', 'thanks', 'what'
        ]
        
        for message in test_messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should be non-empty string
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Should not be too short (at least 10 characters)
            assert len(response) >= 10, \
                f"Response too short for '{message}': {response}"
            
            # Note: Unclear responses intentionally contain example placeholders like [drug name]
            # to show users how to ask questions. This is acceptable and helpful.
    
    def test_case_insensitivity(self):
        """Test that handler works with different cases"""
        test_cases = [
            ('hi', 'Hi', 'HI'),
            ('hello', 'Hello', 'HELLO'),
            ('bye', 'Bye', 'BYE'),
            ('help', 'Help', 'HELP')
        ]
        
        for lower, title, upper in test_cases:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            
            response_lower = conversational_handler.handle(lower, intent)
            response_title = conversational_handler.handle(title, intent)
            response_upper = conversational_handler.handle(upper, intent)
            
            # All should return valid responses
            assert isinstance(response_lower, str) and len(response_lower) > 0
            assert isinstance(response_title, str) and len(response_title) > 0
            assert isinstance(response_upper, str) and len(response_upper) > 0
    
    def test_punctuation_handling(self):
        """Test that handler handles punctuation correctly"""
        test_messages = [
            ('hi', 'hi!', 'hi?', 'hi '),
            ('hello', 'hello!', 'hello?', 'hello '),
            ('bye', 'bye!', 'bye?', 'bye ')
        ]
        
        for messages in test_messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            
            for message in messages:
                response = conversational_handler.handle(message, intent)
                
                # Should return valid response regardless of punctuation
                assert isinstance(response, str)
                assert len(response) > 0
    
    def test_no_medical_analysis_in_conversational_responses(self):
        """Test that conversational responses don't contain medical analysis"""
        conversational_messages = [
            'hi', 'hello', 'bye', 'thanks', 'how are you', 'who are you', 'help'
        ]
        
        for message in conversational_messages:
            intent = Intent(type="conversational", confidence=0.95, extracted_drugs=[])
            response = conversational_handler.handle(message, intent)
            
            # Should not contain medical analysis terms
            medical_terms = ['gnn', 'risk score', 'interaction analysis', 'verdict:', 'do not add']
            has_medical = any(term in response.lower() for term in medical_terms)
            assert not has_medical, \
                f"Conversational response should not contain medical analysis for '{message}', got: {response}"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running ConversationalHandler tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
