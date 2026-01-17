# Design Document: Backend Improvements

## Overview

This design addresses two critical backend improvements for the Medicine Assistant application:

1. **Unified Drug Interaction Analysis**: Consolidate the drug interaction check logic so that both Quick Check (non-logged-in users) and Emergency Check (logged-in users) use the same GNN + RAG + LLM pipeline for consistent, accurate results.

2. **Conversational Chatbot Intelligence**: Implement intent classification to enable the chatbot to distinguish between casual conversation (greetings, small talk) and medical queries (drug interactions, health questions), providing appropriate responses for each.

## Architecture

### Current State Analysis

**Quick Check (`/api/quick-check`)**:
- Currently uses only GNN model for risk prediction
- Does NOT use RAG system for interaction lookup
- Does NOT use LLM for explanation generation
- Returns basic templated responses
- Results in inconsistent quality compared to Emergency Check

**Emergency Check (`/emergency-check`)**:
- Uses GNN model for risk prediction
- Uses RAG system to lookup documented interactions
- Uses LLM (via `ask_local_llm`) for detailed explanations
- Provides comprehensive, personalized responses

**Chatbot (`/ask_assistant`)**:
- Currently assumes ALL messages are medical queries
- Directly processes every message through `get_holistic_context` + `ask_local_llm`
- No intent classification
- Cannot handle casual conversation

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                           │
│  (Quick Check Modal, Emergency Check, Chatbot Interface)    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                          │
│  /api/quick-check  │  /emergency-check  │  /ask_assistant   │
└────────────┬───────┴──────────┬─────────┴──────────┬────────┘
             │                  │                     │
             │                  │                     ▼
             │                  │          ┌──────────────────┐
             │                  │          │ Intent Classifier│
             │                  │          └────┬────────┬────┘
             │                  │               │        │
             │                  │          casual│   medical
             │                  │               │        │
             │                  │               ▼        │
             │                  │     ┌──────────────┐  │
             │                  │     │Conversational│  │
             │                  │     │   Handler    │  │
             │                  │     └──────────────┘  │
             │                  │                       │
             └──────────────────┴───────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │  Interaction Engine      │
                  │  (Unified Analysis Core) │
                  └────────┬─────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐     ┌─────────┐
    │   GNN   │      │   RAG   │     │   LLM   │
    │  Model  │      │ System  │     │ Service │
    └─────────┘      └─────────┘     └─────────┘
```

## Components and Interfaces

### 1. Interaction Engine (New Unified Component)

**Purpose**: Centralized service for drug interaction analysis using GNN + RAG + LLM pipeline.

**Interface**:
```python
class InteractionEngine:
    def __init__(self, gnn_model, drug_map, rag_system, llm_service):
        """Initialize with required dependencies"""
        
    def analyze_interaction(
        self,
        new_drug: str,
        existing_drugs: List[str],
        patient_profile: Optional[Dict] = None,
        dosage_info: Optional[Dict] = None
    ) -> InteractionResult:
        """
        Analyze drug interactions using GNN + RAG + LLM
        
        Args:
            new_drug: The drug being added
            existing_drugs: List of existing drug names
            patient_profile: Optional patient information (age, conditions, etc.)
            dosage_info: Optional dosage details (amount, unit, frequency)
            
        Returns:
            InteractionResult containing:
                - gnn_risk: float (0-100)
                - rag_interactions: List[Dict]
                - llm_explanation: str
                - verdict: str ("SAFE TO ADD" | "CAUTION ADVISED" | "DO NOT ADD")
                - can_add: bool
                - dosage_validation: Dict
        """
```

**Implementation Details**:
- Accepts both single drug and multiple drug scenarios
- Handles missing patient profile gracefully (anonymous users)
- Performs GNN prediction for all drug pairs
- Queries RAG system for documented interactions
- Builds comprehensive context for LLM
- Returns standardized result format

### 2. Intent Classifier (New Component)

**Purpose**: Determine whether a user message is conversational or medical in nature.

**Interface**:
```python
class IntentClassifier:
    def classify(self, message: str) -> Intent:
        """
        Classify user message intent
        
        Args:
            message: User's input text
            
        Returns:
            Intent object with:
                - type: "conversational" | "medical"
                - confidence: float (0-1)
                - extracted_drugs: List[str] (for medical intent)
        """
```

**Classification Strategy**:

**Conversational Patterns** (Rule-based):
- Greetings: `hi`, `hello`, `hey`, `good morning`, `good afternoon`
- Farewells: `bye`, `goodbye`, `see you`, `thanks`, `thank you`
- Small talk: `how are you`, `what's up`, `how's it going`
- Questions about bot: `who are you`, `what can you do`, `help`

**Medical Patterns** (Rule-based + Drug Name Detection):
- Drug name mentions: Check against drug_map keys
- Medical keywords: `take`, `medication`, `drug`, `interaction`, `safe`, `combine`, `dosage`, `side effect`
- Question patterns: `can I take`, `is it safe`, `what about`, `should I`

**Implementation Approach**:
```python
def classify(self, message: str) -> Intent:
    message_lower = message.lower().strip()
    
    # Check conversational patterns first
    conversational_patterns = [
        r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?]*$',
        r'^(bye|goodbye|see you|thanks|thank you)[\s!?]*$',
        r'^how are you',
        r'^what\'?s up',
        r'^who are you',
        r'^what can you do',
        r'^help[\s!?]*$'
    ]
    
    for pattern in conversational_patterns:
        if re.match(pattern, message_lower):
            return Intent(type="conversational", confidence=0.95, extracted_drugs=[])
    
    # Check for drug names
    extracted_drugs = self._extract_drug_names(message)
    
    # Check for medical keywords
    medical_keywords = ['take', 'medication', 'drug', 'interaction', 'safe', 
                       'combine', 'dosage', 'side effect', 'prescription']
    has_medical_keywords = any(keyword in message_lower for keyword in medical_keywords)
    
    # Determine intent
    if extracted_drugs or has_medical_keywords:
        return Intent(type="medical", confidence=0.9, extracted_drugs=extracted_drugs)
    
    # Default to conversational for ambiguous cases
    return Intent(type="conversational", confidence=0.6, extracted_drugs=[])
```

### 3. Conversational Handler (New Component)

**Purpose**: Generate appropriate responses for conversational intents.

**Interface**:
```python
class ConversationalHandler:
    def handle(self, message: str, intent: Intent) -> str:
        """
        Generate conversational response
        
        Args:
            message: User's input text
            intent: Classified intent
            
        Returns:
            Appropriate conversational response
        """
```

**Response Templates**:
```python
CONVERSATIONAL_RESPONSES = {
    'greeting': [
        "Hi! 👋 I'm your AI Health Assistant. I can help you check drug interactions and answer medication questions. What would you like to know?",
        "Hello! 👋 I'm here to help with your medication questions. How can I assist you today?",
        "Hey there! 👋 I'm Dr. MediBot, your health assistant. Ask me about drug interactions or medication safety!"
    ],
    'farewell': [
        "Take care! 👋 Feel free to come back anytime you have medication questions.",
        "Goodbye! Stay healthy and don't hesitate to reach out if you need help. 👋",
        "See you later! Remember, I'm always here to help with your medication questions. 👋"
    ],
    'how_are_you': [
        "I'm doing great, thanks for asking! 😊 I'm here and ready to help with your medication questions. What would you like to know?",
        "I'm functioning perfectly! 🤖 More importantly, how can I help you with your health questions today?"
    ],
    'who_are_you': [
        "I'm Dr. MediBot, your AI health assistant! 🤖 I specialize in checking drug interactions and providing medication safety information. I use advanced AI models to analyze potential risks when combining medications. How can I help you today?",
        "I'm an AI-powered health assistant designed to help you understand drug interactions and medication safety. I analyze your medications using machine learning and medical databases to keep you safe. What would you like to know?"
    ],
    'help': [
        "I can help you with:\n\n✅ Checking drug interactions\n✅ Analyzing medication safety\n✅ Providing dosage information\n✅ Explaining side effects\n\nJust ask me something like: 'Can I take aspirin with ibuprofen?' or 'Is paracetamol safe?'",
        "Here's what I can do for you:\n\n💊 Check if two or more drugs are safe to take together\n📊 Analyze interaction risks using AI\n⚠️ Warn you about potential dangers\n📋 Provide detailed safety information\n\nTry asking: 'Can I combine [drug1] and [drug2]?'"
    ],
    'unclear': [
        "I'm not sure I understand. Are you asking about a specific medication or drug interaction? Feel free to ask me something like: 'Can I take aspirin with ibuprofen?'",
        "Could you clarify what you'd like to know? I'm here to help with medication questions and drug interactions. Try asking about specific drugs!"
    ]
}
```

### 4. Modified Chatbot Endpoint

**Updated Flow**:
```python
@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    """AI Assistant endpoint with intent classification"""
    if 'patient_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        message = data.get('question', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Classify intent
        intent = intent_classifier.classify(message)
        
        # Handle based on intent
        if intent.type == "conversational":
            response = conversational_handler.handle(message, intent)
            return jsonify({
                'response': response,
                'intent': 'conversational',
                'timestamp': datetime.now().isoformat()
            })
        
        # Medical intent - use Interaction Engine
        conn = get_db_connection()
        patient_data = conn.execute('SELECT * FROM patients WHERE id = ?', 
                                   (session['patient_id'],)).fetchone()
        patient = dict(patient_data)
        existing_meds = conn.execute('SELECT * FROM medications WHERE patient_id = ?', 
                                    (session['patient_id'],)).fetchall()
        conn.close()
        
        # Extract drug from message or use existing meds
        if intent.extracted_drugs:
            new_drug = intent.extracted_drugs[0]
            existing_drugs = [med['drug_name'] for med in existing_meds]
        else:
            # Fallback: treat message as drug name
            new_drug = message
            existing_drugs = [med['drug_name'] for med in existing_meds]
        
        # Use Interaction Engine
        result = interaction_engine.analyze_interaction(
            new_drug=new_drug,
            existing_drugs=existing_drugs,
            patient_profile=patient
        )
        
        return jsonify({
            'response': result.llm_explanation,
            'verdict': result.verdict,
            'gnn_risk': result.gnn_risk,
            'intent': 'medical',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Chatbot error: {e}")
        return jsonify({'error': 'Failed to process message'}), 500
```

### 5. Updated Quick Check Endpoint

**Modified to use Interaction Engine**:
```python
@app.route('/api/quick-check', methods=['POST'])
def quick_check():
    """Quick interaction check using unified Interaction Engine"""
    data = request.json
    drugs = data.get('drugs', [])
    
    if len(drugs) < 1:
        return jsonify({'error': 'At least one drug is required'}), 400
    
    # Single drug case
    if len(drugs) == 1:
        return jsonify({
            'gnn_risk': 0,
            'verdict': 'SAFE TO ADD',
            'ai_response': f'{drugs[0]} appears safe when taken alone. Always consult your healthcare provider.',
            'can_add': True
        })
    
    # Multiple drugs - use Interaction Engine
    new_drug = drugs[0]
    existing_drugs = drugs[1:]
    
    result = interaction_engine.analyze_interaction(
        new_drug=new_drug,
        existing_drugs=existing_drugs,
        patient_profile=None  # Anonymous user
    )
    
    return jsonify({
        'gnn_risk': result.gnn_risk,
        'verdict': result.verdict,
        'ai_response': result.llm_explanation,
        'can_add': result.can_add,
        'interactions': result.rag_interactions,
        'dosage_validation': result.dosage_validation
    })
```

### 6. Updated Emergency Check Endpoint

**Modified to use Interaction Engine**:
```python
@app.route('/emergency-check', methods=['POST'])
def emergency_check():
    """Emergency check using unified Interaction Engine"""
    try:
        data = request.json
        if not data or 'drug1' not in data or 'drug2' not in data:
            return jsonify({'status': 'UNSAFE', 'reason': 'Invalid request'}), 400
        
        drug1 = data['drug1'].strip()
        drug2 = data['drug2'].strip()
        
        if not drug1 or not drug2:
            return jsonify({'status': 'UNSAFE', 'reason': 'Both drug names required'}), 400
        
        # Use Interaction Engine
        result = interaction_engine.analyze_interaction(
            new_drug=drug1,
            existing_drugs=[drug2],
            patient_profile=None  # Anonymous emergency check
        )
        
        # Map verdict to status
        status_map = {
            'SAFE TO ADD': 'SAFE',
            'CAUTION ADVISED': 'CAUTION',
            'DO NOT ADD': 'UNSAFE'
        }
        status = status_map.get(result.verdict, 'UNSAFE')
        
        return jsonify({
            'status': status,
            'response': result.llm_explanation,
            'gnn_risk': result.gnn_risk,
            'drug1': drug1,
            'drug2': drug2,
            'interaction': result.rag_interactions[0] if result.rag_interactions else None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Emergency check error: {e}")
        return jsonify({
            'status': 'UNSAFE',
            'response': 'Unable to perform check. Please consult a healthcare professional.',
            'error': str(e)
        }), 500
```

## Data Models

### Intent
```python
@dataclass
class Intent:
    type: str  # "conversational" | "medical"
    confidence: float  # 0.0 to 1.0
    extracted_drugs: List[str]  # Drug names found in message
```

### InteractionResult
```python
@dataclass
class InteractionResult:
    gnn_risk: float  # 0-100
    rag_interactions: List[Dict]  # Documented interactions from RAG
    llm_explanation: str  # Human-readable explanation
    verdict: str  # "SAFE TO ADD" | "CAUTION ADVISED" | "DO NOT ADD"
    can_add: bool  # True if safe to add
    dosage_validation: Dict  # Dosage safety information
    timestamp: str  # ISO format timestamp
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Interaction Engine Consistency

*For any* drug combination and patient profile, calling the Interaction Engine multiple times with the same inputs should produce the same GNN risk score, RAG interactions, and verdict (LLM explanation may vary slightly due to temperature, but verdict should be consistent).

**Validates: Requirements 1.1, 1.2, 2.5**

### Property 2: Quick Check Uses Full Pipeline

*For any* drug list submitted to Quick Check, the response should include GNN risk prediction, RAG interaction lookup, and LLM-generated explanation (not templated responses).

**Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4**

### Property 3: Emergency Check Uses Full Pipeline

*For any* two drugs submitted to Emergency Check, the response should include GNN risk prediction, RAG interaction lookup, and LLM-generated explanation.

**Validates: Requirements 1.2, 2.1, 2.2, 2.3, 2.4**

### Property 4: Conversational Intent Detection

*For any* greeting message (hi, hello, hey, good morning, etc.), the Intent Classifier should identify it as conversational intent with high confidence (>0.9).

**Validates: Requirements 3.1, 3.3**

### Property 5: Medical Intent Detection

*For any* message containing drug names from the drug_map, the Intent Classifier should identify it as medical intent.

**Validates: Requirements 3.2, 3.4**

### Property 6: Conversational Response Appropriateness

*For any* message classified as conversational intent, the Chatbot should NOT invoke the Interaction Engine and should return a conversational response.

**Validates: Requirements 3.3, 4.1, 4.2, 4.3, 4.4**

### Property 7: Medical Query Processing

*For any* message classified as medical intent, the Chatbot should invoke the Interaction Engine and return medical analysis results.

**Validates: Requirements 3.4, 5.1, 5.2, 5.3**

### Property 8: Drug Name Extraction

*For any* message containing valid drug names, the Intent Classifier should extract those drug names correctly.

**Validates: Requirements 5.1**

### Property 9: Standardized Response Format

*For any* interaction analysis (Quick Check, Emergency Check, or Chatbot medical query), the response should contain the required fields: gnn_risk, verdict, explanation, and can_add.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

### Property 10: Error Handling Graceful Degradation

*For any* component failure (GNN, RAG, or LLM), the system should return a partial result or fallback response rather than crashing.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

### Property 11: Anonymous User Support

*For any* Quick Check or Emergency Check request, the system should function correctly without patient profile information.

**Validates: Requirements 1.1, 1.2, 2.1**

### Property 12: Verdict Consistency with Risk Score

*For any* interaction analysis, if GNN risk > 70, verdict should be "DO NOT ADD"; if risk 30-70, verdict should be "CAUTION ADVISED"; if risk < 30, verdict should be "SAFE TO ADD".

**Validates: Requirements 1.3, 7.2**

## Error Handling

### GNN Model Failures
- **Scenario**: GNN model fails to load or predict
- **Handling**: Log error, set gnn_risk to 0, continue with RAG + LLM
- **User Message**: "AI risk prediction unavailable, relying on database lookup"

### RAG System Failures
- **Scenario**: RAG system cannot find interactions
- **Handling**: Return empty interactions list, continue with GNN + LLM
- **User Message**: "No documented interactions found in database"

### LLM Service Failures
- **Scenario**: LLM API is unavailable or times out
- **Handling**: Use fallback response generator based on GNN + RAG results
- **User Message**: Generated from template using available data

### Invalid Drug Names
- **Scenario**: Drug name not in drug_map
- **Handling**: Attempt fuzzy matching, if no match, inform user
- **User Message**: "Drug '[name]' not found. Please check spelling or try another name."

### Empty Message
- **Scenario**: User sends empty or whitespace-only message
- **Handling**: Return error with helpful prompt
- **User Message**: "Please enter a message or question about medications."

### Timeout Handling
- **Scenario**: Interaction Engine takes too long (>5 seconds)
- **Handling**: Return timeout error with retry instructions
- **User Message**: "Request timed out. Please try again."

## Testing Strategy

### Unit Tests

**Interaction Engine Tests**:
- Test with single drug (should return low risk)
- Test with known interacting drugs (should return high risk)
- Test with missing patient profile (should work)
- Test with invalid drug names (should handle gracefully)
- Test dosage validation integration

**Intent Classifier Tests**:
- Test greeting patterns (should return conversational)
- Test farewell patterns (should return conversational)
- Test messages with drug names (should return medical)
- Test messages with medical keywords (should return medical)
- Test ambiguous messages (should default to conversational)
- Test drug name extraction accuracy

**Conversational Handler Tests**:
- Test greeting responses (should be welcoming)
- Test farewell responses (should be polite)
- Test help responses (should list capabilities)
- Test unclear message responses (should ask for clarification)

**Endpoint Integration Tests**:
- Test Quick Check with 2 drugs (should use Interaction Engine)
- Test Emergency Check with 2 drugs (should use Interaction Engine)
- Test Chatbot with greeting (should return conversational response)
- Test Chatbot with medical query (should use Interaction Engine)

### Property-Based Tests

Each property test should run a minimum of 100 iterations with randomized inputs.

**Property 1 Test**: Generate random drug pairs, call Interaction Engine multiple times, verify consistency.

**Property 2 Test**: Generate random drug lists, call Quick Check, verify response contains GNN + RAG + LLM data.

**Property 3 Test**: Generate random drug pairs, call Emergency Check, verify response contains GNN + RAG + LLM data.

**Property 4 Test**: Generate greeting variations, verify Intent Classifier returns conversational with high confidence.

**Property 5 Test**: Generate messages with random drug names from drug_map, verify Intent Classifier returns medical.

**Property 6 Test**: Generate conversational messages, call Chatbot, verify no Interaction Engine invocation.

**Property 7 Test**: Generate medical messages, call Chatbot, verify Interaction Engine invocation.

**Property 8 Test**: Generate messages with embedded drug names, verify extraction accuracy.

**Property 9 Test**: Call all endpoints with random inputs, verify response format consistency.

**Property 10 Test**: Simulate component failures, verify graceful degradation.

**Property 11 Test**: Call Quick Check and Emergency Check without patient profile, verify functionality.

**Property 12 Test**: Generate interactions with various risk scores, verify verdict matches risk level.

### Testing Framework

Use **pytest** for Python testing with **Hypothesis** for property-based testing.

**Example Property Test**:
```python
from hypothesis import given, strategies as st
import pytest

# Feature: backend-improvements, Property 4: Conversational Intent Detection
@given(greeting=st.sampled_from(['hi', 'hello', 'hey', 'good morning', 'good afternoon']))
def test_conversational_intent_detection(greeting):
    """For any greeting message, Intent Classifier should return conversational intent"""
    intent = intent_classifier.classify(greeting)
    assert intent.type == "conversational"
    assert intent.confidence > 0.9
    assert len(intent.extracted_drugs) == 0
```

### Manual Testing Checklist

- [ ] Quick Check with 2 drugs shows GNN risk, RAG interactions, LLM explanation
- [ ] Emergency Check with 2 drugs shows GNN risk, RAG interactions, LLM explanation
- [ ] Chatbot responds to "hi" with greeting (no medical analysis)
- [ ] Chatbot responds to "can I take aspirin with ibuprofen" with medical analysis
- [ ] Chatbot responds to "thank you" with farewell
- [ ] Quick Check and Emergency Check return same quality results
- [ ] All endpoints handle invalid inputs gracefully
- [ ] LLM failure triggers fallback response
- [ ] GNN failure allows system to continue with RAG + LLM

## Performance Considerations

### Caching Strategy
- Cache GNN model in memory (already implemented)
- Cache drug_map for fast lookups (already implemented)
- Consider caching frequent drug pair results (future optimization)

### Response Time Targets
- Intent Classification: < 50ms
- Interaction Engine (with all components): < 5 seconds
- Conversational Response: < 100ms

### Concurrent Request Handling
- Flask handles concurrent requests via threading
- GNN model is thread-safe (read-only after loading)
- RAG system is thread-safe (read-only DataFrame)
- LLM API calls may have rate limits (handle with retry logic)

## Implementation Notes

### Code Organization
```
app.py
├── [Existing code]
├── class InteractionEngine
├── class IntentClassifier  
├── class ConversationalHandler
├── interaction_engine = InteractionEngine(...)  # Global instance
├── intent_classifier = IntentClassifier(...)    # Global instance
├── conversational_handler = ConversationalHandler()  # Global instance
├── [Modified endpoints]
│   ├── /api/quick-check (updated)
│   ├── /emergency-check (updated)
│   └── /ask_assistant (updated)
└── [Existing endpoints]
```

### Backward Compatibility
- All existing endpoints maintain their URL paths
- Response formats remain compatible with frontend
- New fields added to responses (not removed)
- Frontend can continue using existing code

### Deployment Considerations
- No database schema changes required
- No new dependencies required (uses existing libraries)
- Can be deployed as drop-in replacement
- Existing GNN model and data files remain unchanged
