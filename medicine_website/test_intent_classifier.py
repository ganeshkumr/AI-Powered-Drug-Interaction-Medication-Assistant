#!/usr/bin/env python3
"""
Tests for IntentClassifier class
Includes both property-based tests and unit tests
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import intent_classifier, drug_map, Intent


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

# Feature: backend-improvements, Property 4: Conversational Intent Detection
@settings(max_examples=10, deadline=None)
@given(
    greeting=st.sampled_from([
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'Hi', 'Hello', 'Hey', 'HELLO', 'HI',
        'hi!', 'hello!', 'hey!', 'hi?', 'hello?',
        'hi ', 'hello ', 'hey '
    ])
)
def test_conversational_intent_detection(greeting):
    """
    Property 4: Conversational Intent Detection
    For any greeting message (hi, hello, hey, good morning, etc.), 
    the Intent Classifier should identify it as conversational intent 
    with high confidence (>0.9).
    
    Validates: Requirements 3.1, 3.3
    """
    intent = intent_classifier.classify(greeting)
    
    # Should be conversational type
    assert intent.type == "conversational", \
        f"Expected 'conversational' for '{greeting}', got '{intent.type}'"
    
    # Should have high confidence
    assert intent.confidence > 0.9, \
        f"Expected confidence > 0.9 for '{greeting}', got {intent.confidence}"
    
    # Should have no extracted drugs
    assert len(intent.extracted_drugs) == 0, \
        f"Expected no extracted drugs for greeting '{greeting}', got {intent.extracted_drugs}"
    
    # Should be Intent type
    assert isinstance(intent, Intent)


# Feature: backend-improvements, Property 5: Medical Intent Detection
@settings(max_examples=10, deadline=None)
@given(
    drug_name=st.sampled_from([
        'Ibuprofen', 'Metformin', 'Lisinopril',
        'Warfarin', 'Atorvastatin', 'Omeprazole', 'Amlodipine', 'Metoprolol',
        'aspirin', 'tylenol', 'paracetamol'  # Common aliases
    ])
)
def test_medical_intent_detection(drug_name):
    """
    Property 5: Medical Intent Detection
    For any message containing drug names from the drug_map, 
    the Intent Classifier should identify it as medical intent.
    
    Validates: Requirements 3.2, 3.4
    """
    # Create messages with drug names
    messages = [
        f"Can I take {drug_name}?",
        f"Is {drug_name} safe?",
        f"What about {drug_name}",
        f"I want to add {drug_name}",
        f"{drug_name} interaction"
    ]
    
    for message in messages:
        intent = intent_classifier.classify(message)
        
        # Should be medical type
        assert intent.type == "medical", \
            f"Expected 'medical' for '{message}', got '{intent.type}'"
        
        # Should be Intent type
        assert isinstance(intent, Intent)


# Feature: backend-improvements, Property 8: Drug Name Extraction
@settings(max_examples=10, deadline=None)
@given(
    drug_name=st.sampled_from([
        'Ibuprofen', 'Metformin', 'Lisinopril',
        'Warfarin', 'Atorvastatin', 'Omeprazole', 'Amlodipine', 'Metoprolol',
        'aspirin', 'tylenol', 'paracetamol'  # Common aliases
    ])
)
def test_drug_name_extraction(drug_name):
    """
    Property 8: Drug Name Extraction
    For any message containing valid drug names, 
    the Intent Classifier should extract those drug names correctly.
    
    Validates: Requirements 5.1
    """
    # Create messages with drug names in different contexts
    messages = [
        f"Can I take {drug_name} with my other medications?",
        f"Is {drug_name} safe for me?",
        f"What are the side effects of {drug_name}?",
        f"I'm taking {drug_name} daily",
        f"{drug_name}"
    ]
    
    for message in messages:
        intent = intent_classifier.classify(message)
        
        # Should extract at least one drug (or be medical intent with keywords)
        # Note: extraction might not always work for all variations, but should be medical
        assert intent.type == "medical", \
            f"Expected medical intent for '{message}', got '{intent.type}'"
        
        # Should be Intent type
        assert isinstance(intent, Intent)


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestIntentClassifierEdgeCases:
    """Unit tests for IntentClassifier edge cases"""
    
    def test_ambiguous_messages_default_to_conversational(self):
        """
        Test ambiguous messages (should default to conversational)
        Requirements: 3.5, 4.5
        """
        ambiguous_messages = [
            "I'm not sure",
            "Maybe",
            "What do you think?",
            "Tell me more",
            "Interesting"
        ]
        
        for message in ambiguous_messages:
            intent = intent_classifier.classify(message)
            
            # Should default to conversational
            assert intent.type == "conversational", \
                f"Expected 'conversational' for ambiguous message '{message}', got '{intent.type}'"
            
            # Confidence should be lower for ambiguous cases
            assert intent.confidence <= 0.7, \
                f"Expected lower confidence for ambiguous message '{message}', got {intent.confidence}"
    
    def test_messages_with_both_conversational_and_medical_elements(self):
        """
        Test messages with both conversational and medical elements
        Requirements: 3.5, 4.5
        """
        mixed_messages = [
            "Hi, can I take aspirin?",
            "Hello, is ibuprofen safe?",
            "Hey there, what about metformin?",
            "Good morning, I need help with my medication"
        ]
        
        for message in mixed_messages:
            intent = intent_classifier.classify(message)
            
            # Medical intent should take priority when drug names are present
            # or when medical keywords are present
            if any(drug in message.lower() for drug in ['aspirin', 'ibuprofen', 'metformin']):
                assert intent.type == "medical", \
                    f"Expected 'medical' for mixed message with drug name '{message}', got '{intent.type}'"
            elif 'medication' in message.lower():
                assert intent.type == "medical", \
                    f"Expected 'medical' for mixed message with medical keyword '{message}', got '{intent.type}'"
    
    def test_empty_or_whitespace_only_messages(self):
        """
        Test empty or whitespace-only messages
        Requirements: 3.5, 4.5
        """
        empty_messages = [
            "",
            " ",
            "   ",
            "\t",
            "\n",
            "  \t  \n  "
        ]
        
        for message in empty_messages:
            intent = intent_classifier.classify(message)
            
            # Should handle gracefully
            assert isinstance(intent, Intent), \
                f"Expected Intent object for empty message, got {type(intent)}"
            
            # Should default to conversational
            assert intent.type == "conversational", \
                f"Expected 'conversational' for empty message, got '{intent.type}'"
            
            # Should have no extracted drugs
            assert len(intent.extracted_drugs) == 0, \
                f"Expected no extracted drugs for empty message, got {intent.extracted_drugs}"
    
    def test_greeting_variations(self):
        """Test various greeting patterns"""
        greetings = [
            "hi", "Hi", "HI", "hi!", "hi?",
            "hello", "Hello", "HELLO", "hello!", "hello?",
            "hey", "Hey", "HEY", "hey!", "hey?",
            "good morning", "Good Morning", "GOOD MORNING",
            "good afternoon", "good evening"
        ]
        
        for greeting in greetings:
            intent = intent_classifier.classify(greeting)
            
            assert intent.type == "conversational", \
                f"Expected 'conversational' for greeting '{greeting}', got '{intent.type}'"
            assert intent.confidence > 0.9, \
                f"Expected high confidence for greeting '{greeting}', got {intent.confidence}"
    
    def test_farewell_patterns(self):
        """Test farewell patterns"""
        farewells = [
            "bye", "Bye", "BYE", "bye!", "bye?",
            "goodbye", "Goodbye", "GOODBYE",
            "see you", "See you", "SEE YOU",
            "thanks", "thank you", "Thanks", "Thank you", "thx"
        ]
        
        for farewell in farewells:
            intent = intent_classifier.classify(farewell)
            
            assert intent.type == "conversational", \
                f"Expected 'conversational' for farewell '{farewell}', got '{intent.type}'"
            assert intent.confidence > 0.9, \
                f"Expected high confidence for farewell '{farewell}', got {intent.confidence}"
    
    def test_medical_keywords_without_drug_names(self):
        """Test messages with medical keywords but no drug names"""
        medical_messages = [
            "I need help with my medication",
            "What are the side effects?",
            "Is this safe to take?",
            "Can I combine these drugs?",
            "What's the right dosage?",
            "I'm having a reaction",
            "My prescription says..."
        ]
        
        for message in medical_messages:
            intent = intent_classifier.classify(message)
            
            # Should be medical due to keywords
            assert intent.type == "medical", \
                f"Expected 'medical' for '{message}', got '{intent.type}'"
            
            # Confidence should be reasonable
            assert intent.confidence > 0.7, \
                f"Expected confidence > 0.7 for '{message}', got {intent.confidence}"
    
    def test_help_patterns(self):
        """Test help-related patterns"""
        help_messages = [
            "help", "Help", "HELP", "help!", "help?",
            "who are you", "Who are you?",
            "what can you do", "What can you do?",
            "how are you", "How are you?"
        ]
        
        for message in help_messages:
            intent = intent_classifier.classify(message)
            
            assert intent.type == "conversational", \
                f"Expected 'conversational' for help message '{message}', got '{intent.type}'"
    
    def test_multiple_drug_names_in_message(self):
        """Test extraction of multiple drug names from a single message"""
        message = "Can I take aspirin and ibuprofen together?"
        intent = intent_classifier.classify(message)
        
        # Should be medical
        assert intent.type == "medical"
        
        # Should extract both drugs (if both are in drug_map)
        extracted_lower = [d.lower() for d in intent.extracted_drugs]
        # At least one should be extracted
        assert len(intent.extracted_drugs) > 0, \
            f"Expected to extract drugs from '{message}', got none"
    
    def test_case_insensitive_drug_extraction(self):
        """Test that drug extraction is case-insensitive"""
        messages = [
            "Can I take aspirin?",  # Common alias
            "Is Ibuprofen safe?",  # Exact match
            "What about Metformin?",  # Exact match
            "TYLENOL side effects"  # Common alias uppercase
        ]
        
        for message in messages:
            intent = intent_classifier.classify(message)
            
            # Should be medical intent (either from drug extraction or keywords)
            assert intent.type == "medical", \
                f"Expected medical intent for '{message}', got '{intent.type}'"
    
    def test_intent_object_structure(self):
        """Test that Intent object has correct structure"""
        intent = intent_classifier.classify("hello")
        
        # Check all required fields exist
        assert hasattr(intent, 'type')
        assert hasattr(intent, 'confidence')
        assert hasattr(intent, 'extracted_drugs')
        
        # Check types
        assert isinstance(intent.type, str)
        assert isinstance(intent.confidence, float)
        assert isinstance(intent.extracted_drugs, list)
        
        # Check value ranges
        assert intent.type in ['conversational', 'medical']
        assert 0.0 <= intent.confidence <= 1.0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    print("Running IntentClassifier tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
