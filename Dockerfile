FROM python:3.11-slim

WORKDIR /app

# Pre-install CPU-only PyTorch to reduce download size and build time
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY data/ ./data
COPY embedding.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0","--port", "8000"]