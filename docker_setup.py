"""
Optional Docker setup for containerized deployment
"""

dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
'''

docker_compose_content = '''
version: '3.8'
services:
  ts-viz:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
'''

def create_docker_files():
    """Create Docker configuration files"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✓ Docker files created")
    print("To run with Docker:")
    print("  docker-compose up --build")

if __name__ == "__main__":
    create_docker_files()