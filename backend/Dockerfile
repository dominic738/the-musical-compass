# Base Python image
FROM python:3.11-slim

# Prevents Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . /app

# Expose port for FastAPI (Cloud Run injects $PORT, default 8080)
EXPOSE 8080

# Run FastAPI with uvicorn, honouring $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}