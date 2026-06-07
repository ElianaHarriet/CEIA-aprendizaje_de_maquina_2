FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

COPY app.py /app/app.py

EXPOSE 8800

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]