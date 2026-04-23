import os
import time
from datetime import datetime, timedelta
import shutil

def cleanup_old_files():
    """Очистка старых файлов"""
    uploads_dir = 'uploads'
    results_dir = 'results'
    
    # Время жизни файлов в часах
    max_age_hours = 24
    
    current_time = datetime.now()
    
    for folder in [uploads_dir, results_dir]:
        if not os.path.exists(folder):
            continue
            
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            
            if os.path.isfile(item_path):
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                    age = current_time - mtime
                    
                    if age > timedelta(hours=max_age_hours):
                        os.remove(item_path)
                        print(f"Удален старый файл: {item_path}")
                except Exception as e:
                    print(f"Ошибка при обработке файла {item_path}: {e}")

if __name__ == '__main__':
    print("Сервис очистки запущен")
    print("Очистка будет выполняться каждые 6 часов")
    
    while True:
        try:
            cleanup_old_files()
        except Exception as e:
            print(f"Ошибка при выполнении очистки: {e}")
        
        # Ожидание 6 часов
        time.sleep(6 * 3600)