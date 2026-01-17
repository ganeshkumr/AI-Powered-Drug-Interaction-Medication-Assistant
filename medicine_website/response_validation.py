"""
Response format validation functions for backend improvements
"""

from typing import Dict
from app import InteractionResult


def validate_interaction_result(result: InteractionResult) -> bool:
    """
    Validate that InteractionResult has all required fields with correct types
    
    Args:
        result: InteractionResult to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check all required fields exist
        required_fields = ['gnn_risk', 'rag_interactions', 'llm_explanation', 
                          'verdict', 'can_add', 'dosage_validation', 'timestamp']
        
        for field in required_fields:
            if not hasattr(result, field):
                print(f"[VALIDATION ERROR] Missing field: {field}")
                return False
        
        # Validate field types
        if not isinstance(result.gnn_risk, (int, float)):
            print(f"[VALIDATION ERROR] gnn_risk must be numeric, got {type(result.gnn_risk)}")
            return False
        
        if not isinstance(result.rag_interactions, list):
            print(f"[VALIDATION ERROR] rag_interactions must be list, got {type(result.rag_interactions)}")
            return False
        
        if not isinstance(result.llm_explanation, str):
            print(f"[VALIDATION ERROR] llm_explanation must be string, got {type(result.llm_explanation)}")
            return False
        
        if not isinstance(result.verdict, str):
            print(f"[VALIDATION ERROR] verdict must be string, got {type(result.verdict)}")
            return False
        
        if not isinstance(result.can_add, bool):
            print(f"[VALIDATION ERROR] can_add must be bool, got {type(result.can_add)}")
            return False
        
        if not isinstance(result.dosage_validation, dict):
            print(f"[VALIDATION ERROR] dosage_validation must be dict, got {type(result.dosage_validation)}")
            return False
        
        if not isinstance(result.timestamp, str):
            print(f"[VALIDATION ERROR] timestamp must be string, got {type(result.timestamp)}")
            return False
        
        # Validate verdict values
        valid_verdicts = ['SAFE TO ADD', 'CAUTION ADVISED', 'DO NOT ADD']
        if result.verdict not in valid_verdicts:
            print(f"[VALIDATION ERROR] Invalid verdict: {result.verdict}")
            return False
        
        # Validate gnn_risk range
        if not (0 <= result.gnn_risk <= 100):
            print(f"[VALIDATION ERROR] gnn_risk must be 0-100, got {result.gnn_risk}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[VALIDATION ERROR] Exception during validation: {e}")
        return False


def validate_response_format(response_dict: Dict, endpoint_type: str) -> bool:
    """
    Validate that response dictionary has standardized format for the endpoint
    
    Args:
        response_dict: Response dictionary to validate
        endpoint_type: Type of endpoint ('quick_check', 'emergency_check', 'chatbot_medical', 'chatbot_conversational')
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if endpoint_type == 'quick_check':
            # Quick Check must have: gnn_risk, verdict, ai_response, can_add
            required = ['gnn_risk', 'verdict', 'ai_response', 'can_add']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Quick Check missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            if not isinstance(response_dict['verdict'], str):
                return False
            if not isinstance(response_dict['ai_response'], str):
                return False
            if not isinstance(response_dict['can_add'], bool):
                return False
            
        elif endpoint_type == 'emergency_check':
            # Emergency Check must have: status, response, gnn_risk, drug1, drug2
            required = ['status', 'response', 'gnn_risk', 'drug1', 'drug2']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Emergency Check missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['status'], str):
                return False
            if not isinstance(response_dict['response'], str):
                return False
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            
            # Validate status values
            valid_statuses = ['SAFE', 'CAUTION', 'UNSAFE']
            if response_dict['status'] not in valid_statuses:
                print(f"[VALIDATION ERROR] Invalid status: {response_dict['status']}")
                return False
            
        elif endpoint_type == 'chatbot_medical':
            # Chatbot medical must have: response, verdict, gnn_risk, intent
            required = ['response', 'verdict', 'gnn_risk', 'intent']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Chatbot medical missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['response'], str):
                return False
            if not isinstance(response_dict['verdict'], str):
                return False
            if not isinstance(response_dict['gnn_risk'], (int, float)):
                return False
            if response_dict['intent'] != 'medical':
                print(f"[VALIDATION ERROR] Intent must be 'medical', got {response_dict['intent']}")
                return False
            
        elif endpoint_type == 'chatbot_conversational':
            # Chatbot conversational must have: response, intent
            required = ['response', 'intent']
            for field in required:
                if field not in response_dict:
                    print(f"[VALIDATION ERROR] Chatbot conversational missing field: {field}")
                    return False
            
            # Validate types
            if not isinstance(response_dict['response'], str):
                return False
            if response_dict['intent'] != 'conversational':
                print(f"[VALIDATION ERROR] Intent must be 'conversational', got {response_dict['intent']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[VALIDATION ERROR] Exception during response validation: {e}")
        return False
