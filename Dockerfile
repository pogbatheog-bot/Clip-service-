FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "900", "app:app"]
