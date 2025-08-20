
FROM pytorch/pytorch:2.8.0-cpu


WORKDIR /app


COPY backend/ ./backend
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000


CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]