#!/bin/bash

echo "Обновление QLF Анализатора..."

# Останавливаем текущие контейнеры
docker-compose down

# Удаляем старые образы
docker image prune -f

# Пересобираем контейнеры
docker-compose build --no-cache

# Запускаем заново
docker-compose up -d

echo "Обновление завершено!"