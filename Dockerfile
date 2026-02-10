FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STATE_DIR=/tmp/openclaw-state
ENV PORT=10000

EXPOSE 10000

CMD ["python", "server.py"]
