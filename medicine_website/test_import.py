import sys
import traceback

try:
    import app
    print("Import successful")
    print("validate_interaction_result in dir(app):", "validate_interaction_result" in dir(app))
    print("validate_response_format in dir(app):", "validate_response_format" in dir(app))
    
    # Try to access them directly
    try:
        func1 = app.validate_interaction_result
        print("validate_interaction_result accessible:", True)
    except AttributeError as e:
        print("validate_interaction_result accessible:", False, str(e))
    
    try:
        func2 = app.validate_response_format
        print("validate_response_format accessible:", True)
    except AttributeError as e:
        print("validate_response_format accessible:", False, str(e))
        
except Exception as e:
    print("Import failed:")
    traceback.print_exc()
