FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RELAY_HOST=0.0.0.0 \
    RELAY_PORT=8765 \
    RELAY_CONSOLE_PORT=8766

WORKDIR /app

RUN pip install --no-cache-dir "websockets>=13.0" "aiohttp>=3.9" \
    && useradd --system --uid 65532 --no-create-home --shell /usr/sbin/nologin relay

COPY --chown=65532:65532 server.py /app/server.py

USER 65532:65532

EXPOSE 8765 8766

CMD ["python", "/app/server.py"]
