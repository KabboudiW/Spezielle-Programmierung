FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser -m appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser . .
USER appuser
ENV PYTHONPYCACHEPREFIX=/home/appuser/.cache/pycache
EXPOSE 8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]