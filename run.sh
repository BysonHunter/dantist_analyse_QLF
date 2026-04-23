#!/bin/bash

# Скрипт для запуска QLF Анализатора

echo "Запуск QLF Анализатора зубной эмали..."

# Проверяем установлен ли Docker
if ! command -v docker &> /dev/null; then
    echo "Docker не установлен. Пожалуйста, установите Docker сначала."
    exit 1
fi

# Проверяем установлен ли Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose не установлен. Устанавливаем..."
    # Для Linux
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Создаем необходимые директории
mkdir -p uploads results artifacts

# Проверяем наличие файла requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "Файл requirements.txt не найден. Создаем..."
    cat > requirements.txt << EOF
Flask==2.3.3
Flask-CORS==4.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
Werkzeug==2.3.7
EOF
fi

# Запускаем контейнеры
echo "Запуск контейнеров..."
docker-compose up -d

echo ""
echo "=============================================="
echo "QLF Анализатор успешно запущен!"
echo ""
echo "Доступные адреса:"
echo "  - Веб-интерфейс: http://localhost:5000"
echo "  - REST API: http://localhost:5000/analyze"
echo ""
echo "Команды управления:"
echo "  Остановить: docker-compose down"
echo "  Просмотр логов: docker-compose logs -f"
echo "  Перезапустить: docker-compose restart"
echo "=============================================="