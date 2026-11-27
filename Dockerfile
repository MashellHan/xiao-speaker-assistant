FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE config.example.yaml ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && useradd -m -u 1000 xsa \
    && mkdir -p /data /app/models \
    && chown -R xsa:xsa /data /app

USER xsa

VOLUME ["/data", "/app/models"]

EXPOSE 4397 4399

ENTRYPOINT ["xsa"]
CMD ["--config", "/data/config.yaml", "serve", "--mcp-host", "0.0.0.0"]
