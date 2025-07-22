
import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting Time Series Visualization Server...")
    print("Visit http://localhost:8000 to view the interface")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )