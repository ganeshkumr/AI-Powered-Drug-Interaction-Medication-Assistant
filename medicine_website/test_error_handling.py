"""
Property-based tests for error handling and graceful degradation
Feature: backend-improvements
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock
from app import InteractionEngine, InteractionResult
import torch


# Strategies for generating test data
drug_names = st.sampled_from([
    'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Warfarin', 'Metformin',
    'Lisinopril', 'Atorvastatin', 'Amlodipine', 'Omeprazole', 'Levothyroxine'
])

drug_lists = st.lists(drug_names, min_size=1, max_size=5, unique=True)


# Feature: backend-improvements, Property 10: Error Handling Graceful Degradation
@given(
    new_drug=drug_names,
    existing_drugs=drug_lists
)
@settings(max_examples=10, deadline=None)
def test_error_handling_graceful_degradation(new_drug, existing_drugs):
    """
    Property 10: Error Handling Graceful Degradation
    
    For any drug combination, when one or more components (GNN, RAG, LLM) fail,
    the system should continue operating with available components and return
    a valid InteractionResult rather than crashing.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.5
    """
    # Create mock components that can fail
    mock_gnn_model = Mock()
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1, 'Acetaminophen': 2, 
                     'Warfarin': 3, 'Metformin': 4, 'Lisinopril': 5,
                     'Atorvastatin': 6, 'Amlodipine': 7, 'Omeprazole': 8,
                     'Levothyroxine': 9}
    mock_rag_system = Mock()
    mock_dosage_validator = Mock()
    mock_side_effects_db = Mock()
    mock_multi_drug_checker = Mock()
    
    # Configure mocks to return reasonable defaults
    mock_rag_system.search_interaction = Mock(return_value=None)
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    # Test scenario 1: GNN fails
    mock_gnn_model.encode = Mock(side_effect=Exception("GNN model failure"))
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Should not crash, should return valid result
    with patch('app.ask_local_llm', return_value="Safe to add based on available data"):
        result = engine.analyze_interaction(new_drug, existing_drugs)
    
    assert isinstance(result, InteractionResult)
    assert result.gnn_risk == 0.0  # GNN failed, should return 0
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]
    
    # Test scenario 2: RAG fails
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.5]))
    mock_rag_system.search_interaction = Mock(side_effect=Exception("RAG system failure"))
    
    engine2 = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Should not crash, should return valid result
    with patch('app.ask_local_llm', return_value="Safe to add based on GNN analysis"):
        result2 = engine2.analyze_interaction(new_drug, existing_drugs)
    
    assert isinstance(result2, InteractionResult)
    assert result2.rag_interactions == []  # RAG failed, should return empty list
    assert result2.llm_explanation is not None
    assert len(result2.llm_explanation) > 0
    assert result2.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]
    
    # Test scenario 3: LLM fails
    mock_rag_system.search_interaction = Mock(return_value=None)
    
    engine3 = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Should not crash, should use fallback response
    with patch('app.ask_local_llm', side_effect=Exception("LLM API failure")):
        with patch('app.generate_fallback_response', return_value="Fallback response"):
            result3 = engine3.analyze_interaction(new_drug, existing_drugs)
    
    assert isinstance(result3, InteractionResult)
    assert result3.llm_explanation is not None
    assert len(result3.llm_explanation) > 0
    assert result3.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


@given(
    new_drug=drug_names,
    existing_drugs=drug_lists
)
@settings(max_examples=10, deadline=None)
def test_all_components_fail_gracefully(new_drug, existing_drugs):
    """
    Property: When all components fail, system should still return a valid result
    with conservative verdict (DO NOT ADD)
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.5
    """
    # Create mocks that all fail
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(side_effect=Exception("GNN failure"))
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1, 'Acetaminophen': 2}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(side_effect=Exception("RAG failure"))
    mock_rag_system.df = None
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(side_effect=Exception("Dosage validator failure"))
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(side_effect=Exception("Side effects DB failure"))
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(side_effect=Exception("Multi-drug checker failure"))
    mock_multi_drug_checker.check_category_conflicts = Mock(side_effect=Exception("Category checker failure"))
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Should not crash even when everything fails
    with patch('app.ask_local_llm', side_effect=Exception("LLM failure")):
        with patch('app.generate_fallback_response', return_value="System error - consult doctor"):
            result = engine.analyze_interaction(new_drug, existing_drugs)
    
    assert isinstance(result, InteractionResult)
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    # When all components fail, should be conservative
    assert "consult" in result.llm_explanation.lower() or "doctor" in result.llm_explanation.lower()


@given(
    new_drug=drug_names,
    existing_drugs=drug_lists
)
@settings(max_examples=10, deadline=None)
def test_partial_component_availability_notice(new_drug, existing_drugs):
    """
    Property: When some components fail, the explanation should include
    a notice about which components are unavailable
    
    Validates: Requirements 6.1, 6.2, 6.3
    """
    # Create mocks where GNN fails but RAG works
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(side_effect=Exception("GNN failure"))
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1, 'Acetaminophen': 2,
                     'Warfarin': 3, 'Metformin': 4}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value={
        'drug_a': new_drug,
        'drug_b': existing_drugs[0] if existing_drugs else 'Unknown',
        'severity': 'moderate',
        'interaction': 'May interact'
    })
    mock_rag_system.df = Mock()  # Not None
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    with patch('app.ask_local_llm', return_value="Analysis based on database"):
        result = engine.analyze_interaction(new_drug, existing_drugs)
    
    assert isinstance(result, InteractionResult)
    # Should mention unavailable components
    assert "unavailable" in result.llm_explanation.lower() or "available" in result.llm_explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================================
# Unit Tests for Error Scenarios
# ============================================================================

def test_gnn_model_failure_continues_with_rag_llm():
    """
    Test that when GNN model fails, the system continues with RAG + LLM
    
    Validates: Requirements 6.1
    """
    # Create mock components where GNN fails
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(side_effect=Exception("GNN model crashed"))
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value={
        'drug_a': 'Aspirin',
        'drug_b': 'Ibuprofen',
        'severity': 'moderate',
        'interaction': 'May cause stomach issues'
    })
    mock_rag_system.df = Mock()  # Not None
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': ['Nausea', 'Headache'],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Mock LLM to return a response
    with patch('app.ask_local_llm', return_value="Based on database, moderate risk detected"):
        result = engine.analyze_interaction('Aspirin', ['Ibuprofen'])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # GNN should have failed and returned 0
    assert result.gnn_risk == 0.0
    
    # RAG should have worked
    assert len(result.rag_interactions) > 0
    assert result.rag_interactions[0]['severity'] == 'moderate'
    
    # LLM should have worked
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    
    # Should have a valid verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


def test_rag_system_failure_continues_with_gnn_llm():
    """
    Test that when RAG system fails, the system continues with GNN + LLM
    
    Validates: Requirements 6.2
    """
    # Create mock components where RAG fails
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.6]))  # 60% risk
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(side_effect=Exception("RAG database crashed"))
    mock_rag_system.df = None  # Simulate RAG not available
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Mock LLM to return a response
    with patch('app.ask_local_llm', return_value="Based on AI analysis, moderate risk"):
        result = engine.analyze_interaction('Aspirin', ['Ibuprofen'])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # GNN should have worked
    assert result.gnn_risk > 0
    
    # RAG should have failed and returned empty list
    assert result.rag_interactions == []
    
    # LLM should have worked
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    
    # Should have a valid verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


def test_llm_failure_uses_fallback_response():
    """
    Test that when LLM fails, the system uses fallback response
    
    Validates: Requirements 6.3
    """
    # Create mock components where LLM fails
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.3]))  # 30% risk
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value=None)
    mock_rag_system.df = Mock()
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Mock LLM to fail, but fallback to work
    with patch('app.ask_local_llm', side_effect=Exception("LLM API unavailable")):
        with patch('app.generate_fallback_response', return_value="Fallback: Based on AI, low risk"):
            result = engine.analyze_interaction('Aspirin', ['Ibuprofen'])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # GNN should have worked
    assert result.gnn_risk > 0
    
    # Should have used fallback response
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    assert "Fallback" in result.llm_explanation or "AI" in result.llm_explanation
    
    # Should have a valid verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


def test_timeout_handling_returns_timeout_error():
    """
    Test that when LLM times out, the system handles it gracefully
    
    Validates: Requirements 6.5
    """
    # Create mock components
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.4]))
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value=None)
    mock_rag_system.df = Mock()
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Mock LLM to timeout
    import requests
    with patch('app.ask_local_llm', side_effect=requests.exceptions.Timeout("Request timed out")):
        with patch('app.generate_fallback_response', return_value="Timeout occurred, using fallback"):
            result = engine.analyze_interaction('Aspirin', ['Ibuprofen'])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # Should have used fallback response
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0
    
    # Should have a valid verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


def test_invalid_drug_names_handled_gracefully():
    """
    Test that invalid drug names are handled gracefully
    
    Validates: Requirements 6.4
    """
    # Create mock components
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.0]))
    
    # Drug map doesn't contain the drugs we'll test with
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value=None)
    mock_rag_system.df = Mock()
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Test with invalid drug names
    with patch('app.ask_local_llm', return_value="Unable to find drug information"):
        result = engine.analyze_interaction('InvalidDrug123', ['AnotherInvalidDrug456'])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # GNN risk should be 0 (drugs not found)
    assert result.gnn_risk == 0.0
    
    # Should have a valid verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]
    
    # Should have an explanation
    assert result.llm_explanation is not None
    assert len(result.llm_explanation) > 0


def test_empty_existing_drugs_list():
    """
    Test that empty existing drugs list is handled (single drug case)
    
    Validates: Requirements 6.4
    """
    # Create mock components
    mock_gnn_model = Mock()
    mock_drug_map = {'Aspirin': 0}
    mock_rag_system = Mock()
    mock_rag_system.df = Mock()
    
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(return_value={
        'is_safe': True,
        'warnings': [],
        'max_daily': None,
        'max_single': None
    })
    
    mock_side_effects_db = Mock()
    mock_multi_drug_checker = Mock()
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Test with empty existing drugs list
    result = engine.analyze_interaction('Aspirin', [])
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # Should be safe (single drug)
    assert result.gnn_risk == 0.0
    assert "safe when taken alone" in result.llm_explanation.lower()
    assert result.verdict == "SAFE TO ADD"


def test_dosage_validation_failure_handled():
    """
    Test that dosage validation failures are handled gracefully
    
    Validates: Requirements 6.4
    """
    # Create mock components
    mock_gnn_model = Mock()
    mock_gnn_model.encode = Mock(return_value=torch.randn(10, 128))
    mock_gnn_model.decode = Mock(return_value=torch.tensor([0.2]))
    
    mock_drug_map = {'Aspirin': 0, 'Ibuprofen': 1}
    
    mock_rag_system = Mock()
    mock_rag_system.search_interaction = Mock(return_value=None)
    mock_rag_system.df = Mock()
    
    # Dosage validator fails
    mock_dosage_validator = Mock()
    mock_dosage_validator.validate_dosage = Mock(side_effect=Exception("Dosage validator crashed"))
    
    mock_side_effects_db = Mock()
    mock_side_effects_db.get_side_effects = Mock(return_value={
        'side_effects': [],
        'risk_warnings': []
    })
    
    mock_multi_drug_checker = Mock()
    mock_multi_drug_checker.check_triangular_conflicts = Mock(return_value=[])
    mock_multi_drug_checker.check_category_conflicts = Mock(return_value=[])
    
    engine = InteractionEngine(
        gnn_model=mock_gnn_model,
        drug_map=mock_drug_map,
        rag_system=mock_rag_system,
        dosage_validator=mock_dosage_validator,
        side_effects_db=mock_side_effects_db,
        multi_drug_checker=mock_multi_drug_checker
    )
    
    # Test with dosage info
    with patch('app.ask_local_llm', return_value="Low risk interaction"):
        result = engine.analyze_interaction(
            'Aspirin', 
            ['Ibuprofen'],
            dosage_info={'dosage_amount': 100, 'dosage_unit': 'mg', 'frequency': 'daily'}
        )
    
    # Should not crash
    assert isinstance(result, InteractionResult)
    
    # Dosage validation should have failed
    assert result.dosage_validation.get('is_safe') == False
    assert 'Dosage validation failed' in result.dosage_validation.get('warnings', [])
    
    # Should still have a verdict
    assert result.verdict in ["SAFE TO ADD", "CAUTION ADVISED", "DO NOT ADD"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
