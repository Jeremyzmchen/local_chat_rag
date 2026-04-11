FROM python:3.11-slim

WORKDIR /app

# System deps for faiss / pdfminer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

# Copy source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
