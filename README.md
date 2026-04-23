# QLF Анализатор зубной эмали

Веб-приложение для анализа деминерализации зубной эмали с использованием метода количественной светоиндуцированной флуоресценции (QLF).

## Основные функции

1. Загрузка QLF изображений зубов
2. Интерактивный выбор области интереса (ROI)
3. Автоматическая сегментация и анализ всех зубов на снимке
4. Визуализация поражений с цветовой кодировкой
5. Детальная статистика QLF метрик
6. Сохранение результатов в различных форматах

## Установка и запуск

### Требования
- Python 3.8+
- Современный веб-браузер

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск приложения локально

```bash
python app.py
```

Приложение будет доступно по адресу: `http://localhost:5000`.

## Docker: сборка и запуск

### Требования

- Docker Desktop (или Docker Engine + Compose Plugin)
- Включенный Docker daemon

### Вариант 1: через Docker Compose (рекомендуется)

1. Перейти в корень проекта:

```bash
cd /path/to/dantist_analise_QLF
```

2. Собрать и запустить сервисы:

```bash
docker compose up --build
```

3. Открыть в браузере:

`http://localhost:5000`

4. Остановить:

```bash
docker compose down
```

### Вариант 2: через docker build + docker run

1. Собрать образ:

```bash
docker build -t qlf-analyzer:latest .
```

2. Запустить контейнер:

```bash
docker run --name qlf-analyzer -p 5000:5000 ^
  -v %cd%/uploads:/app/uploads ^
  -v %cd%/results:/app/results ^
  -v %cd%/artifacts:/app/artifacts ^
  qlf-analyzer:latest
```

Для Linux/macOS замените `^` на `\` и `%cd%` на `$(pwd)`.

### Полезные команды

```bash
docker compose logs -f
docker compose ps
docker compose down -v
```
