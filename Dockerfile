FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XSA_HOME=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && useradd -m -u 1000 xsa \
    && mkdir -p /data /app/models \
    && chown -R xsa:xsa /data /app

USER xsa

VOLUME ["/data", "/app/models"]

EXPOSE 4397 4399

ENTRYPOINT ["xsa"]
CMD ["serve", "--config", "/data/config.yaml"]
