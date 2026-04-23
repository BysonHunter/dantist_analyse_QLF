import cv2
import numpy as np

def apply_color_palette_to_image(image, analysis_data, color_palette):
    """
    Применяет цветовую палитру к изображению на основе данных анализа.
    
    Args:
        image: Исходное изображение (BGR)
        analysis_data: Данные анализа из функции tooth_Analise
        color_palette: Палитра цветов из colorpalette.py
    
    Returns:
        Изображение с раскрашенными областями поражения эмали
    """
    if image is None:
        return None
    
    # Создаем копию изображения
    result_image = image.copy()
    
    # Получаем данные анализа
    if not analysis_data or len(analysis_data) < 12:
        print("⚠️ Недостаточно данных анализа для раскраски")
        return result_image
    
    # Распаковываем данные анализа
    # q0, q1, q2, q3, q4, mean_deltaR, changing_ratio, max_changing_ratio, 
    # mean_deltaF, avg_fluorescence_loss, max_fluorescence_loss
    # Но нам нужны оригинальные данные из tooth_Analise
    
    print(f"📊 Применение цветовой палитры: {len(color_palette)} цветов")
    
    # Пример раскраски по зонам (замените на реальную логику из tooth_Analise)
    # Здесь создаем упрощенную визуализацию
    
    # Создаем маску для разных уровней поражения
    height, width = image.shape[:2]
    
    # Разделяем изображение на зоны для демонстрации
    zone_height = height // 6
    
    # Применяем цвета из палитры
    for i, color in enumerate(color_palette):
        if i < 6:  # Используем первые 6 цветов
            y_start = i * zone_height
            y_end = (i + 1) * zone_height
            
            # Применяем цвет к зоне (в BGR формате, как в OpenCV)
            # color из pallete в формате (B, G, R)
            result_image[y_start:y_end, :] = np.array(color, dtype=np.uint8)
            
            # Добавляем текст с описанием
            cv2.putText(result_image, f'Zone {i+1}', (10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result_image


def create_detailed_visualization(original_image, mask, contours, color_palette):
    """
    Создает детализированную визуализацию с раскраской по уровням поражения.
    
    Args:
        original_image: Оригинальное изображение
        mask: Маска области зуба
        contours: Контуры зуба
        color_palette: Цветовая палитра
    
    Returns:
        Изображение с детальной раскраской
    """
    if original_image is None:
        return None
    
    # Создаем изображение для визуализации
    viz_image = original_image.copy()
    
    # Если есть маска, применяем ее
    if mask is not None:
        # Создаем цветную маску
        colored_mask = np.zeros_like(original_image)
        
        # Раскрашиваем маску в разные цвета в зависимости от интенсивности
        mask_normalized = mask.astype(float) / 255.0
        
        # Применяем градиент цветов из палитры
        for i in range(len(color_palette) - 1):
            color1 = np.array(color_palette[i], dtype=float)
            color2 = np.array(color_palette[i + 1], dtype=float)
            
            # Создаем диапазон значений маски для этого цвета
            lower_threshold = i / len(color_palette)
            upper_threshold = (i + 1) / len(color_palette)
            
            # Находим пиксели в этом диапазоне
            mask_range = (mask_normalized >= lower_threshold) & (mask_normalized < upper_threshold)
            
            if mask_range.any():
                # Интерполируем цвет
                factor = (mask_normalized[mask_range] - lower_threshold) / (upper_threshold - lower_threshold)
                for c in range(3):  # Для каждого канала BGR
                    colored_mask[mask_range, c] = color1[c] + (color2[c] - color1[c]) * factor
        
        # Наложение цветной маски на оригинальное изображение
        alpha = 0.6  # Прозрачность
        viz_image = cv2.addWeighted(colored_mask, alpha, viz_image, 1 - alpha, 0)
    
    # Рисуем контуры
    if contours:
        cv2.drawContours(viz_image, contours, -1, (0, 255, 0), 2)
    
    # Добавляем легенду
    legend_height = 30
    legend_width = 200
    legend = np.zeros((legend_height * len(color_palette), legend_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(color_palette):
        y_start = i * legend_height
        y_end = (i + 1) * legend_height
        legend[y_start:y_end, :] = color
        
        # Текст легенды
        cv2.putText(legend, f'Level {i+1}', (10, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Объединяем изображение с легендой
    if viz_image.shape[0] > legend.shape[0]:
        # Создаем пустое пространство справа для легенды
        padding = np.zeros((viz_image.shape[0], legend.shape[1], 3), dtype=np.uint8)
        padding[:legend.shape[0], :] = legend
        viz_image = np.hstack([viz_image, padding])
    
    return viz_image