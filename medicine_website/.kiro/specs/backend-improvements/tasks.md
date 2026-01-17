# Implementation Plan: Backend Improvements

## Overview

This implementation plan addresses two critical backend improvements: unifying the drug interaction analysis pipeline across all endpoints (Quick Check, Emergency Check, and Chatbot) and implementing conversational intelligence in the chatbot to distinguish between casual conversation and medical queries.

## Tasks

- [x] 1. Create InteractionEngine class for unified drug analysis
  - Create new `InteractionEngine` class in app.py
  - Implement `analyze_interaction()` method that combines GNN + RAG + LLM
  - Handle both single and multiple drug scenarios
  - Support optional patient profile and dosage information
  - Return standardized `InteractionResult` with all required fields
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 1.1 Write property test for InteractionEngine consistency
  - **Property 1: Interaction Engine Consistency**
  - **Validates: Requirements 1.1, 1.2, 2.5**

- [x] 1.2 Write unit tests for InteractionEngine edge cases
  - Test with single drug (should return low risk)
  - Test with invalid drug names (should handle gracefully)
  - Test with missing patient profile (should work)
  - _Requirements: 1.1, 1.2, 6.4_

- [x] 2. Create IntentClassifier class for message classification
  - Create new `IntentClassifier` class in app.py
  - Implement `classify()` method with rule-based pattern matching
  - Add conversational pattern detection (greetings, farewells, small talk)
  - Add medical pattern detection (drug names, medical keywords)
  - Implement drug name extraction from messages
  - Return `Intent` object with type, confidence, and extracted drugs
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1_

- [x] 2.1 Write property test for conversational intent detection
  - **Property 4: Conversational Intent Detection**
  - **Validates: Requirements 3.1, 3.3**

- [x] 2.2 Write property test for medical intent detection
  - **Property 5: Medical Intent Detection**
  - **Validates: Requirements 3.2, 3.4**

- [x] 2.3 Write property test for drug name extraction
  - **Property 8: Drug Name Extraction**
  - **Validates: Requirements 5.1**

- [x] 2.4 Write unit tests for IntentClassifier edge cases
  - Test ambiguous messages (should default to conversational)
  - Test messages with both conversational and medical elements
  - Test empty or whitespace-only messages
  - _Requirements: 3.5, 4.5_

- [x] 3. Create ConversationalHandler class for casual responses
  - Create new `ConversationalHandler` class in app.py
  - Define response templates for greetings, farewells, help, etc.
  - Implement `handle()` method that selects appropriate response
  - Add randomization to avoid repetitive responses
  - Ensure responses are friendly and guide users toward medical queries
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 3.1 Write property test for conversational response appropriateness
  - **Property 6: Conversational Response Appropriateness**
  - **Validates: Requirements 3.3, 4.1, 4.2, 4.3, 4.4**

- [x] 3.2 Write unit tests for ConversationalHandler responses
  - Test greeting responses (should be welcoming)
  - Test farewell responses (should be polite)
  - Test help responses (should list capabilities)
  - Test unclear message responses (should ask for clarification)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4. Initialize global instances of new components
  - Create global `interaction_engine` instance after loading GNN model
  - Create global `intent_classifier` instance with drug_map reference
  - Create global `conversational_handler` instance
  - Add error handling for initialization failures
  - _Requirements: 2.1, 3.1, 4.1_

- [x] 5. Update /api/quick-check endpoint to use InteractionEngine
  - Replace current GNN-only logic with InteractionEngine call
  - Handle single drug case (return safe response)
  - For multiple drugs, call `interaction_engine.analyze_interaction()`
  - Map InteractionResult to existing response format
  - Ensure backward compatibility with frontend
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 7.3_

- [x] 5.1 Write property test for Quick Check full pipeline
  - **Property 2: Quick Check Uses Full Pipeline**
  - **Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.4**

- [x] 5.2 Write integration test for Quick Check endpoint
  - Test with 2 drugs (should return GNN + RAG + LLM)
  - Test with single drug (should return safe response)
  - Test with invalid drugs (should handle gracefully)
  - _Requirements: 1.1, 6.4, 7.3_

- [x] 6. Update /emergency-check endpoint to use InteractionEngine
  - Replace current logic with InteractionEngine call
  - Call `interaction_engine.analyze_interaction()` with drug1 and drug2
  - Map InteractionResult verdict to status ("SAFE", "CAUTION", "UNSAFE")
  - Ensure response format matches frontend expectations
  - _Requirements: 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 7.3_

- [x] 6.1 Write property test for Emergency Check full pipeline
  - **Property 3: Emergency Check Uses Full Pipeline**
  - **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 2.4**

- [x] 6.2 Write integration test for Emergency Check endpoint
  - Test with 2 drugs (should return GNN + RAG + LLM)
  - Test with invalid drugs (should handle gracefully)
  - Test status mapping (verdict to status conversion)
  - _Requirements: 1.2, 6.4, 7.3_

- [x] 7. Update /ask_assistant endpoint with intent classification
  - Add intent classification at the start of the endpoint
  - If conversational intent: use ConversationalHandler, return response
  - If medical intent: extract drugs, use InteractionEngine
  - Update response format to include intent type
  - Ensure backward compatibility with existing chatbot UI
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7.1 Write property test for medical query processing
  - **Property 7: Medical Query Processing**
  - **Validates: Requirements 3.4, 5.1, 5.2, 5.3**

- [x] 7.2 Write integration test for Chatbot endpoint
  - Test with greeting (should return conversational response)
  - Test with medical query (should use InteractionEngine)
  - Test with farewell (should return conversational response)
  - Test with drug name in message (should extract and analyze)
  - _Requirements: 3.1, 3.2, 4.1, 5.1, 5.2_

- [x] 8. Checkpoint - Test all endpoints manually
  - Ensure all tests pass, ask the user if questions arise.
  - Manually test Quick Check with 2 drugs
  - Manually test Emergency Check with 2 drugs
  - Manually test Chatbot with greetings and medical queries
  - Verify response formats match frontend expectations

- [x] 9. Add error handling and fallback mechanisms
  - Add try-catch blocks in InteractionEngine for GNN failures
  - Add try-catch blocks in InteractionEngine for RAG failures
  - Add try-catch blocks in InteractionEngine for LLM failures
  - Implement graceful degradation (continue with available components)
  - Add timeout handling for LLM API calls
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9.1 Write property test for error handling graceful degradation
  - **Property 10: Error Handling Graceful Degradation**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.5**

- [x] 9.2 Write unit tests for error scenarios
  - Test GNN model failure (should continue with RAG + LLM)
  - Test RAG system failure (should continue with GNN + LLM)
  - Test LLM failure (should use fallback response)
  - Test timeout handling (should return timeout error)
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [x] 10. Add response format validation
  - Create helper function to validate InteractionResult format
  - Ensure all endpoints return standardized response structure
  - Add response format validation to all three endpoints
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10.1 Write property test for standardized response format
  - **Property 9: Standardized Response Format**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [x] 10.2 Write unit tests for response format validation
  - Test Quick Check response format
  - Test Emergency Check response format
  - Test Chatbot response format (both conversational and medical)
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 11. Add anonymous user support validation
  - Verify InteractionEngine works without patient profile
  - Verify Quick Check works without authentication
  - Verify Emergency Check works without authentication
  - _Requirements: 1.1, 1.2, 2.1_

- [x] 11.1 Write property test for anonymous user support
  - **Property 11: Anonymous User Support**
  - **Validates: Requirements 1.1, 1.2, 2.1**

- [x] 12. Add verdict consistency validation
  - Implement logic to ensure verdict matches risk score
  - Risk > 70: "DO NOT ADD"
  - Risk 30-70: "CAUTION ADVISED"
  - Risk < 30: "SAFE TO ADD"
  - Add validation in InteractionEngine
  - _Requirements: 1.3, 7.2_

- [x] 12.1 Write property test for verdict consistency
  - **Property 12: Verdict Consistency with Risk Score**
  - **Validates: Requirements 1.3, 7.2**

- [x] 13. Final checkpoint - Comprehensive testing
  - Ensure all tests pass, ask the user if questions arise.
  - Run all unit tests
  - Run all property tests
  - Run all integration tests
  - Perform manual testing of all three endpoints
  - Verify frontend compatibility

- [ ] 14. Update documentation and add logging
  - Add docstrings to all new classes and methods
  - Add logging statements for debugging
  - Update code comments for clarity
  - Document any breaking changes (if any)

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end flows
- All endpoints maintain backward compatibility with existing frontend
