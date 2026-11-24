FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    RELAY_HOST=0.0.0.0 \
    RELAY_PORT=8765

WORKDIR /app

RUN pip install --no-cache-dir "websockets>=13.0"

COPY server.py /app/server.py

EXPOSE 8765

CMD ["python", "/app/server.py"]
