# setup_fastapi.py
"""
Setup script to prepare the FastAPI visualization server
"""
import os
import pickle
import torch

def save_training_data_for_api(training_data, filepath="training_data.pkl"):
    """
    Save training data to pickle file for FastAPI to load
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(training_data, f)
        print(f"✓ Training data saved to {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error saving training data: {e}")
        return False

def create_models_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs("models", exist_ok=True)
    print("✓ Models directory created/verified")

def verify_model_files():
    """Verify that required model files exist"""
    required_files = [
        "models/final_model.pth",
        "models/model_after_all_voltage_phases.pth"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✓ All required model files found")
        return True

def setup_fastapi_environment(training_data=None):
    """
    Complete setup for FastAPI visualization server
    """
    print("Setting up FastAPI environment...")
    
    # Create directories
    create_models_directory()
    
    # Save training data if provided
    if training_data is not None:
        save_training_data_for_api(training_data)
    
    # Check model files
    verify_model_files()
    
    print("\n" + "="*50)
    print("FastAPI SETUP COMPLETE")
    print("="*50)
    print("To start the server, run:")
    print("  python fastapi_server.py")
    print("  OR")
    print("  uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000")
    print("\nThen visit: http://localhost:8000")
    print("="*50)