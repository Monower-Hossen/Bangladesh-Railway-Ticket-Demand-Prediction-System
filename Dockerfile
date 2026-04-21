FROM python:3.9-slim

WORKDIR /app

# Only requirements first (cache optimization)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Then copy rest
COPY . .

CMD ["python", "app.py"]