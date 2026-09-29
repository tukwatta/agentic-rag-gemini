# Base image: Linux + Python already installed
FROM python:3.12-slim

# Print logs immediately (so `docker compose logs` shows them live)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first and install them in their own layer.
# Docker caches layers, so editing app.py won't re-download every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app code (NOT .env - secrets must never be baked into an image)
COPY app.py .

# Streamlit's port
EXPOSE 8501

# 0.0.0.0 = listen on all network interfaces, required so the port is reachable
# from outside the container
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
