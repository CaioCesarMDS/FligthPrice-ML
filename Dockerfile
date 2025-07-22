FROM python:3.13.5

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY data/ ./data/
COPY mlruns/ ./mlruns/

CMD ["python", "main.py"]
