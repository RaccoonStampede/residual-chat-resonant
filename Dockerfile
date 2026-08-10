FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 5000

ENV FLASK_PORT=5000 \
    FLASK_DEBUG=False \
    STORAGE_PATH=/app/data \
    DATABASE_URL=sqlite:////app/data/residual.db \
    ENABLE_PERSISTENCE=True

CMD ["python", "web_chat.py"]
