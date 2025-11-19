FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
COPY model.py .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python model.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]