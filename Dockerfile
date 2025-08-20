# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your code
COPY . /app

# Install CPU-only PyTorch + requirements
RUN pip install --no-cache-dir torch==2.8.0 \
    && pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]