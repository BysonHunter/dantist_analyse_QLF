# Официальный образ Python (slim — уже уменьшенный)
FROM python:3.11-slim-bookworm

# Только рантайм-зависимости для OpenCV (без ffmpeg, gcc, libpq — не используются)
# curl — для healthcheck в docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libfreetype6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Единый список зависимостей
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Директории приложения
RUN mkdir -p uploads results artifacts templates

# Код приложения (исключения — в .dockerignore)
COPY . .


# Создаем пользователя для безопасности (опционально, но рекомендуется)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Открываем порт
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]