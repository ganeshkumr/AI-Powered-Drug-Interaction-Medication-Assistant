"""
Property-based tests for response format validation
Feature: backend-improvements, Property 9: Standardized Response Format
Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from app import (
    interaction_engine, 
    intent_classifier, 
    conversational_handler,
    InteractionResult
)
from response_validation import validate_interaction_result, validate_response_format
from datetime import datetime


# Generators for property-based testing
@st.composite
def drug_name_strategy(draw):
    """Generate valid drug names"""
    drugs = ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Lisinopril', 'Metformin', 
             'Atorvastatin', 'Amlodipine', 'Omeprazole', 'Losartan', 'Gabapentin']
    return draw(st.sampled_from(drugs))


@st.composite
def drug_list_strategy(draw):
    """Generate list of 1-3 drugs"""
    drugs = ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Lisinopril', 'Metformin']
    num_drugs = draw(st.integers(min_value=1, max_value=3))
    return draw(st.lists(st.sampled_from(drugs), min_size=num_drugs, max_size=num_drugs, unique=True))


# Property 9: Standardized Response Format
# For any interaction analysis (Quick Check, Emergency Check, or Chatbot medical query),
# the response should contain the required fields: gnn_risk, verdict, explanation, and can_add

@settings(max_examples=20, deadline=None)
@given(
    new_drug=drug_name_strategy(),
    existing_drugs=st.lists(drug_name_strategy(), min_size=0, max_size=2, unique=True)
)
def test_interaction_result_has_standardized_format(new_drug, existing_drugs):
    """
    Property 9: Standardized Response Format
    For any drug combination, InteractionEngine should return a result with all required fields
    Validates: Requirements 7.1, 7.2, 7.3, 7.4
    """
    # Analyze interaction
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None
    )
    
    # Verify result is an InteractionResult
    assert isinstance(result, InteractionResult), "Result must be InteractionResult type"
    
    # Verify all required fields exist
    assert hasattr(result, 'gnn_risk'), "Result must have gnn_risk field"
    assert hasattr(result, 'rag_interactions'), "Result must have rag_interactions field"
    assert hasattr(result, 'llm_explanation'), "Result must have llm_explanation field"
    assert hasattr(result, 'verdict'), "Result must have verdict field"
    assert hasattr(result, 'can_add'), "Result must have can_add field"
    assert hasattr(result, 'dosage_validation'), "Result must have dosage_validation field"
    assert hasattr(result, 'timestamp'), "Result must have timestamp field"
    
    # Verify field types
    assert isinstance(result.gnn_risk, (int, float)), "gnn_risk must be numeric"
    assert isinstance(result.rag_interactions, list), "rag_interactions must be list"
    assert isinstance(result.llm_explanation, str), "llm_explanation must be string"
    assert isinstance(result.verdict, str), "verdict must be string"
    assert isinstance(result.can_add, bool), "can_add must be bool"
    assert isinstance(result.dosage_validation, dict), "dosage_validation must be dict"
    assert isinstance(result.timestamp, str), "timestamp must be string"
    
    # Verify verdict values
    valid_verdicts = ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
    assert result.verdict in valid_verdicts, f"verdict must be one of {valid_verdicts}"
    
    # Verify gnn_risk range
    assert 0 <= result.gnn_risk <= 100, "gnn_risk must be between 0 and 100"
    
    # Verify validation function agrees
    assert validate_interaction_result(result), "validate_interaction_result should return True"


@settings(max_examples=20, deadline=None)
@given(
    new_drug=drug_name_strategy(),
    existing_drugs=st.lists(drug_name_strategy(), min_size=0, max_size=2, unique=True)
)
def test_quick_check_response_format(new_drug, existing_drugs):
    """
    Property 9: Quick Check has standardized response format
    For any drug list, Quick Check response should have required fields
    Validates: Requirements 7.1, 7.3
    """
    # Simulate Quick Check response
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None
    )
    
    # Build Quick Check response format
    response_data = {
        'gnn_risk': result.gnn_risk,
        'verdict': result.verdict,
        'ai_response': result.llm_explanation,
        'can_add': result.can_add,
        'dosage_validation': result.dosage_validation
    }
    
    # Verify required fields
    assert 'gnn_risk' in response_data, "Quick Check must have gnn_risk"
    assert 'verdict' in response_data, "Quick Check must have verdict"
    assert 'ai_response' in response_data, "Quick Check must have ai_response"
    assert 'can_add' in response_data, "Quick Check must have can_add"
    
    # Verify types
    assert isinstance(response_data['gnn_risk'], (int, float)), "gnn_risk must be numeric"
    assert isinstance(response_data['verdict'], str), "verdict must be string"
    assert isinstance(response_data['ai_response'], str), "ai_response must be string"
    assert isinstance(response_data['can_add'], bool), "can_add must be bool"
    
    # Verify validation function agrees
    assert validate_response_format(response_data, 'quick_check'), "Quick Check format validation should pass"


@settings(max_examples=20, deadline=None)
@given(
    drug1=drug_name_strategy(),
    drug2=drug_name_strategy()
)
def test_emergency_check_response_format(drug1, drug2):
    """
    Property 9: Emergency Check has standardized response format
    For any two drugs, Emergency Check response should have required fields
    Validates: Requirements 7.2, 7.3
    """
    # Simulate Emergency Check response
    result = interaction_engine.analyze_interaction(
        new_drug=drug1,
        existing_drugs=[drug2],
        patient_profile=None
    )
    
    # Map verdict to status
    status_map = {
        'SAFE TO ADD': 'SAFE',
        'CAUTION ADVISED': 'CAUTION',
        'DO NOT ADD': 'UNSAFE'
    }
    status = status_map.get(result.verdict, 'UNSAFE')
    
    # Build Emergency Check response format
    response_data = {
        'status': status,
        'response': result.llm_explanation,
        'gnn_risk': result.gnn_risk,
        'drug1': drug1,
        'drug2': drug2,
        'interaction': result.rag_interactions[0] if result.rag_interactions else None,
        'timestamp': result.timestamp
    }
    
    # Verify required fields
    assert 'status' in response_data, "Emergency Check must have status"
    assert 'response' in response_data, "Emergency Check must have response"
    assert 'gnn_risk' in response_data, "Emergency Check must have gnn_risk"
    assert 'drug1' in response_data, "Emergency Check must have drug1"
    assert 'drug2' in response_data, "Emergency Check must have drug2"
    
    # Verify types
    assert isinstance(response_data['status'], str), "status must be string"
    assert isinstance(response_data['response'], str), "response must be string"
    assert isinstance(response_data['gnn_risk'], (int, float)), "gnn_risk must be numeric"
    
    # Verify status values
    valid_statuses = ['SAFE', 'CAUTION', 'UNSAFE']
    assert response_data['status'] in valid_statuses, f"status must be one of {valid_statuses}"
    
    # Verify validation function agrees
    assert validate_response_format(response_data, 'emergency_check'), "Emergency Check format validation should pass"


@settings(max_examples=10, deadline=None)
@given(
    new_drug=drug_name_strategy(),
    existing_drugs=st.lists(drug_name_strategy(), min_size=0, max_size=2, unique=True)
)
def test_chatbot_medical_response_format(new_drug, existing_drugs):
    """
    Property 9: Chatbot medical response has standardized format
    For any medical query, Chatbot should return response with required fields
    Validates: Requirements 7.4
    """
    # Simulate Chatbot medical response
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None
    )
    
    # Build Chatbot medical response format
    response_data = {
        'response': result.llm_explanation,
        'verdict': result.verdict,
        'gnn_risk': result.gnn_risk,
        'intent': 'medical',
        'timestamp': datetime.now().isoformat()
    }
    
    # Verify required fields
    assert 'response' in response_data, "Chatbot medical must have response"
    assert 'verdict' in response_data, "Chatbot medical must have verdict"
    assert 'gnn_risk' in response_data, "Chatbot medical must have gnn_risk"
    assert 'intent' in response_data, "Chatbot medical must have intent"
    
    # Verify types
    assert isinstance(response_data['response'], str), "response must be string"
    assert isinstance(response_data['verdict'], str), "verdict must be string"
    assert isinstance(response_data['gnn_risk'], (int, float)), "gnn_risk must be numeric"
    assert response_data['intent'] == 'medical', "intent must be 'medical'"
    
    # Verify validation function agrees
    assert validate_response_format(response_data, 'chatbot_medical'), "Chatbot medical format validation should pass"


@settings(max_examples=10, deadline=None)
@given(
    greeting=st.sampled_from(['hi', 'hello', 'hey', 'good morning', 'good afternoon'])
)
def test_chatbot_conversational_response_format(greeting):
    """
    Property 9: Chatbot conversational response has standardized format
    For any conversational message, Chatbot should return response with required fields
    Validates: Requirements 7.4
    """
    # Classify intent
    intent = intent_classifier.classify(greeting)
    
    # Get conversational response
    response = conversational_handler.handle(greeting, intent)
    
    # Build Chatbot conversational response format
    response_data = {
        'response': response,
        'intent': 'conversational',
        'timestamp': datetime.now().isoformat()
    }
    
    # Verify required fields
    assert 'response' in response_data, "Chatbot conversational must have response"
    assert 'intent' in response_data, "Chatbot conversational must have intent"
    
    # Verify types
    assert isinstance(response_data['response'], str), "response must be string"
    assert response_data['intent'] == 'conversational', "intent must be 'conversational'"
    
    # Verify validation function agrees
    assert validate_response_format(response_data, 'chatbot_conversational'), "Chatbot conversational format validation should pass"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
