FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and the trained model (see .dockerignore for what is left out)
COPY . .

# Run as a non-root user
RUN useradd --create-home app
USER app

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health', timeout=3)"

CMD ["python", "serve.py"]
