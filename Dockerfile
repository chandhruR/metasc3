FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

ENV PORT=7860
EXPOSE 7860

CMD sh -c "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860}"
