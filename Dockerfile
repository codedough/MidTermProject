FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# (versions match the Requirements section of README.md)
RUN pip install --no-cache-dir numpy==2.5.3 pandas==3.0.5 scikit-learn==1.9.1 flask==3.1.3 gunicorn==26.2.0

# Only what the service needs: code and the trained model
COPY produce_features.py predict.py ./
COPY model/shortage_model.bin model/shortage_model.bin

# Run as a non-root user
RUN useradd --create-home app
USER app

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "predict:app"]
