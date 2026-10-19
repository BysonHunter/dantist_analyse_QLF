import cv2 
import numpy as np
import datetime
import json
try:
    import paths
except ImportError:
    paths = None  # Only make_json requires the application-specific paths module.

def find_conturs(image):
    """
    Находит контуры на изображении.

    Args:
        image: Исходное изображение.

    Returns:
        Кортеж из четырех элементов:
            - Изображение с наложенными контурами.
            - Маска с выделенными контурами.
            - Список контуров.
            - Иерархия контуров.
    """

    # Define HSV color range for tooth
    h_min = np.array((0, 0, 38), np.uint8)
    h_max = np.array((255, 155, 255), np.uint8)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Thresholding for HSV
    hsv_mask = cv2.inRange(hsv, h_min, h_max)

    # Find HSV contours
    contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw HSV contours on the image
    img_with_hsv_conturs = cv2.bitwise_and(image.copy(), image.copy(), mask=hsv_mask)

    # Convert HSV-masked image to grayscale
    gray = cv2.cvtColor(img_with_hsv_conturs, cv2.COLOR_BGR2GRAY)

    # Thresholding for grayscale
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove noise with morphological operation (меньше итераций — быстрее, качество приемлемо)
    kernel = np.ones((3, 3), np.uint8)
    gray_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Find grayscale contours
    contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw grayscale contours on the image
    #img_with_gray_conturs = cv2.bitwise_and(image.copy(), image.copy(), mask=gray_mask)
    img_with_gray_conturs = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    return img_with_gray_conturs, gray_mask, contours, hierarchy


def mean_colors(image):
    """Channel-preserving BGR means over non-black spatial pixels.

    Zero components inside a non-black pixel remain part of that pixel.
    All-black background pixels are excluded. Returns zero for an empty ROI.
    This fixes scalar filtering in the supplied implementation.
    """
    if image.size == 0:
        return 0.0, 0.0, 0.0
    pixels = image[np.any(image != 0, axis=2)]
    if len(pixels) == 0:
        return 0.0, 0.0, 0.0
    return tuple(np.mean(pixels.astype(np.float64), axis=0))


def mean_colors_100(image):
    """Вычисляет средние значения цветов заданных самых ярких пикселей в изображении (векторизовано)."""
    # Только ненулевые пиксели
    mask = np.any(image != 0, axis=2)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 0, 0, 0, []
    pixels = image[rows, cols]
    brightness = np.sum(pixels, axis=1)
    cnt = len(brightness)
    cnt1 = max(1, cnt // 2)
    idx_top = np.argsort(brightness)[::-1][:cnt1]
    top_B = np.mean(pixels[idx_top, 0])
    top_G = np.mean(pixels[idx_top, 1])
    top_R = np.mean(pixels[idx_top, 2])
    lights = [(brightness[k], pixels[k, 0], pixels[k, 1], pixels[k, 2]) for k in idx_top]
    return int(top_B), int(top_G), int(top_R), lights


def tooth_Analise(img, contoursToMask, colors):
    """
    Анализирует изображение зубов, выделяет область челюсти по контуру и рассчитывает метрики.

    Args:
        img: Исходное изображение.
        contoursToMask: Контуры зуба, которые нужно использовать для маскирования.
        colors: Палетка цветов для раскраски пикселей.

    Returns:
        Изображение с наложенной маской, метрики зуба: q0, q1, q2, q3, q4, 
        mean_deltaR, changing_the_ratio_of_red_and_green_fluorescence, 
        max_changing_the_ratio_of_red_and_green_fluorescence, mean_deltaF, 
        average_fluorescence_loss, maximum_fluorescence_loss. 
    """

    # Создаем маску
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contoursToMask, -1, (255, 0, 0), -1, cv2.LINE_AA)

    # Выделяем область зуба
    result_of_mask = cv2.bitwise_and(img, img, mask=mask)

    # Вычисляем средние значения цветов
    mean_B, mean_G, mean_R = mean_colors(result_of_mask)
    # mean_B, mean_G, mean_R, _ = mean_colors_100(result_of_mask)

    # Создаем копию изображения для визуализации
    image_with_mask = img.copy()

    # Вычисляем дельты цвета — итерация только по пикселям маски (ускорение)
    deltas = []
    count_points_in_mask = 0
    cnt_tresh120, cnt_tresh90, cnt_tresh75, cnt_tresh60, cnt_tresh45 = 0, 0, 0, 0, 0
    rows, cols = np.where(mask > 0)

    for idx in range(len(rows)):
        j, i = rows[idx], cols[idx]
        color_result1 = image_with_mask[j, i]
        color_result2 = result_of_mask[j, i]

        if color_result2.all() != 0:
            count_points_in_mask += 1

            if color_result1[2] == color_result2[2] and (color_result1[0] <= mean_B or color_result1[1] <= mean_G):
                deltaB = (mean_B - color_result1[0]) if color_result1[0] < mean_B else 0
                deltaG = (mean_G - color_result1[1]) if color_result1[1] < mean_G else 0
                deltaR = (mean_R - color_result1[2]) if color_result1[2] < mean_R else 0
                deltaR_plus = (color_result1[2] - mean_R) if color_result1[2] > mean_R else 0
                delta1 = deltaG + deltaR

                if delta1 > 120:
                    cnt_tresh120 += 1
                elif 120 >= delta1 > 90:
                    cnt_tresh90 += 1
                elif 90 >= delta1 > 75:
                    color_result1[:] = colors[2]
                    cnt_tresh75 += 1
                elif 75 >= delta1 > 60:
                    color_result1[:] = colors[3]
                    cnt_tresh60 += 1
                elif 60 >= delta1 > 45:
                    color_result1[:] = colors[4]
                    cnt_tresh45 += 1

                if deltaR_plus > 10:
                    color_result1[:] = colors[5]

                deltas.append([deltaB, deltaG, deltaR, delta1, deltaR_plus])

    # Вычисляем средние значения дельт
    mean_delta1 = mean_B + mean_G + mean_R
    cnt_tresh25 = 0

    mean_deltaF = 0
    mean_deltaR = 0
    max_deltaF = 0
    max_deltaR = 0

    for delta in deltas:
        if delta[3] > 45:
            cnt_tresh25 += 1
            mean_deltaF += delta[3]
            max_deltaF = max(max_deltaF, delta[3])

        if delta[3] > 120:
            mean_deltaR += delta[3]
            max_deltaR = max(max_deltaR, delta[3])

    # Проверка на ненулевое количество пикселей
    if cnt_tresh25 == 0:
        return image_with_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Вычисляем метрики
    cnt_all = len(deltas)
    mean_deltaR = round(mean_deltaR / cnt_tresh25, 2)
    changing_the_ratio_of_red_and_green_fluorescence = round(mean_deltaR / mean_delta1, 2)
    max_changing_the_ratio_of_red_and_green_fluorescence = round(max_deltaR / mean_delta1, 2)

    mean_deltaF = round(mean_deltaF / cnt_tresh25, 2)
    average_fluorescence_loss = round(mean_deltaF / mean_delta1, 2)
    maximum_fluorescence_loss = round(max_deltaF / mean_delta1, 2)

    q0 = cnt_tresh25 / cnt_all
    q1 = cnt_tresh45 / cnt_all
    q2 = cnt_tresh60 / cnt_all
    q3 = cnt_tresh75 / cnt_all
    q4 = cnt_tresh90 / cnt_all

    return image_with_mask, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        mean_deltaF, \
        average_fluorescence_loss, maximum_fluorescence_loss


def tooth_Analise_green(img, contoursToMask, colors):
    """
    Анализ по зелёному каналу (аналогично tooth_Analise, метрики по зелёной флуоресценции).
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contoursToMask, -1, (255, 0, 0), -1, cv2.LINE_AA)
    result_of_mask = cv2.bitwise_and(img, img, mask=mask)
    mean_B, mean_G, mean_R = mean_colors(result_of_mask)
    image_with_mask = img.copy()
    deltas = []
    cnt_tresh120, cnt_tresh90, cnt_tresh75, cnt_tresh60, cnt_tresh45 = 0, 0, 0, 0, 0
    rows, cols = np.where(mask > 0)

    for idx in range(len(rows)):
        j, i = rows[idx], cols[idx]
        color_result1 = image_with_mask[j, i]
        color_result2 = result_of_mask[j, i]
        if color_result2.all() != 0:
            if color_result1[1] == color_result2[1] and (color_result1[0] <= mean_B or color_result1[2] <= mean_R):
                deltaB = (mean_B - color_result1[0]) if color_result1[0] < mean_B else 0
                deltaG = (mean_G - color_result1[1]) if color_result1[1] < mean_G else 0
                deltaR = (mean_R - color_result1[2]) if color_result1[2] < mean_R else 0
                deltaG_plus = (color_result1[1] - mean_G) if color_result1[1] > mean_G else 0
                delta1 = deltaG + deltaR
                if delta1 > 120:
                    cnt_tresh120 += 1
                elif 120 >= delta1 > 90:
                    cnt_tresh90 += 1
                elif 90 >= delta1 > 75:
                    color_result1[:] = colors[2]
                    cnt_tresh75 += 1
                elif 75 >= delta1 > 60:
                    color_result1[:] = colors[3]
                    cnt_tresh60 += 1
                elif 60 >= delta1 > 45:
                    color_result1[:] = colors[4]
                    cnt_tresh45 += 1
                if deltaG_plus > 10:
                    color_result1[:] = colors[5]
                deltas.append([deltaB, deltaG, deltaR, delta1, deltaG_plus])

    mean_delta1 = mean_B + mean_G + mean_R
    cnt_tresh25 = 0
    mean_deltaF = 0
    mean_deltaG = 0
    max_deltaF = 0
    max_deltaG = 0
    for delta in deltas:
        if delta[3] > 45:
            cnt_tresh25 += 1
            mean_deltaF += delta[3]
            max_deltaF = max(max_deltaF, delta[3])
        if delta[3] > 120:
            mean_deltaG += delta[3]
            max_deltaG = max(max_deltaG, delta[3])
    if cnt_tresh25 == 0:
        return image_with_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt_all = len(deltas)
    mean_deltaG = round(mean_deltaG / cnt_tresh25, 2)
    changing_the_ratio_of_green_fluorescence = round(mean_deltaG / mean_delta1, 2)
    max_changing_the_ratio_of_green_fluorescence = round(max_deltaG / mean_delta1, 2)
    mean_deltaF = round(mean_deltaF / cnt_tresh25, 2)
    average_fluorescence_loss = round(mean_deltaF / mean_delta1, 2)
    maximum_fluorescence_loss = round(max_deltaF / mean_delta1, 2)
    q0 = cnt_tresh25 / cnt_all
    q1 = cnt_tresh45 / cnt_all
    q2 = cnt_tresh60 / cnt_all
    q3 = cnt_tresh75 / cnt_all
    q4 = cnt_tresh90 / cnt_all
    return image_with_mask, q0, q1, q2, q3, q4, mean_deltaG, \
        changing_the_ratio_of_green_fluorescence, max_changing_the_ratio_of_green_fluorescence, \
        mean_deltaF, average_fluorescence_loss, maximum_fluorescence_loss

        
def select_tooth_Analise(img, contoursToMask, colors):
    """
    Анализирует изображение зуба, выделяет область по контуру и рассчитывает метрики.

    Args:
        img: Исходное изображение.
        contoursToMask: Контуры зуба, которые нужно использовать для маскирования.
        colors: Палетка цветов для раскраски пикселей.

    Returns:
        Изображение с наложенной маской, метрики зуба: q0, q1, q2, q3, q4, 
        mean_deltaR, changing_the_ratio_of_red_and_green_fluorescence, 
        max_changing_the_ratio_of_red_and_green_fluorescence, mean_deltaF, 
        average_fluorescence_loss, maximum_fluorescence_loss. 
    """

    # Создаем маску
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contoursToMask, -1, (255, 0, 0), -1, cv2.LINE_AA)

    # Выделяем область зуба
    result_of_mask = cv2.bitwise_and(img, img, mask=mask)

    # Вычисляем средние значения цветов
    # mean_B, mean_G, mean_R = mean_colors(result_of_mask)
    mean_B, mean_G, mean_R, _ = mean_colors_100(result_of_mask)

    # Создаем копию изображения для визуализации
    image_with_mask = img.copy()

    # Вычисляем дельты цвета — итерация только по пикселям маски (ускорение)
    deltas = []
    count_points_in_mask = 0
    cnt_tresh120, cnt_tresh90, cnt_tresh75, cnt_tresh60, cnt_tresh45 = 0, 0, 0, 0, 0
    rows, cols = np.where(mask > 0)

    for idx in range(len(rows)):
        j, i = rows[idx], cols[idx]
        color_result1 = image_with_mask[j, i]
        color_result2 = result_of_mask[j, i]

        if color_result2.all() != 0:
            count_points_in_mask += 1

            if color_result1[2] == color_result2[2] and (color_result1[0] <= mean_B or color_result1[1] <= mean_G):
                deltaB = (mean_B - color_result1[0]) if color_result1[0] < mean_B else 0
                deltaG = (mean_G - color_result1[1]) if color_result1[1] < mean_G else 0
                deltaR = (mean_R - color_result1[2]) if color_result1[2] < mean_R else 0
                deltaR_plus = (color_result1[2] - mean_R) if color_result1[2] > mean_R else 0
                delta1 = deltaG + deltaR

                if delta1 > 120:
                    color_result1[:] = colors[0]
                    cnt_tresh120 += 1
                elif 120 >= delta1 > 90:
                    color_result1[:] = colors[1]
                    cnt_tresh90 += 1
                elif 90 >= delta1 > 75:
                    color_result1[:] = colors[2]
                    cnt_tresh75 += 1
                elif 75 >= delta1 > 60:
                    color_result1[:] = colors[3]
                    cnt_tresh60 += 1
                elif 60 >= delta1 > 45:
                    color_result1[:] = colors[4]
                    cnt_tresh45 += 1

                if deltaR_plus > 10:
                    color_result1[:] = colors[5]

                deltas.append([deltaB, deltaG, deltaR, delta1, deltaR_plus])

    # Вычисляем средние значения дельт
    mean_delta1 = mean_B + mean_G + mean_R
    cnt_tresh25 = 0

    mean_deltaF = 0
    mean_deltaR = 0
    max_deltaF = 0
    max_deltaR = 0

    for delta in deltas:
        if delta[3] > 45:
            cnt_tresh25 += 1
            mean_deltaF += delta[3]
            max_deltaF = max(max_deltaF, delta[3])

        if delta[3] > 120:
            mean_deltaR += delta[3]
            max_deltaR = max(max_deltaR, delta[3])

    # Проверка на ненулевое количество пикселей
    if cnt_tresh25 == 0:
        return image_with_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Вычисляем метрики
    cnt_all = len(deltas)
    mean_deltaR = round(mean_deltaR / cnt_tresh25, 2)
    changing_the_ratio_of_red_and_green_fluorescence = round(mean_deltaR / mean_delta1, 2)
    max_changing_the_ratio_of_red_and_green_fluorescence = round(max_deltaR / mean_delta1, 2)

    mean_deltaF = round(mean_deltaF / cnt_tresh25, 2)
    average_fluorescence_loss = round(mean_deltaF / mean_delta1, 2)
    maximum_fluorescence_loss = round(max_deltaF / mean_delta1, 2)

    q0 = cnt_tresh25 / cnt_all
    q1 = cnt_tresh45 / cnt_all
    q2 = cnt_tresh60 / cnt_all
    q3 = cnt_tresh75 / cnt_all
    q4 = cnt_tresh90 / cnt_all

    return image_with_mask, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        mean_deltaF, \
        average_fluorescence_loss, maximum_fluorescence_loss


def select_tooth_Analise_green(img, contoursToMask, colors):
    """
    Анализ по зелёному каналу для прямоугольного ROI (аналогично select_tooth_Analise).
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contoursToMask, -1, (255, 0, 0), -1, cv2.LINE_AA)
    result_of_mask = cv2.bitwise_and(img, img, mask=mask)
    mean_B, mean_G, mean_R, _ = mean_colors_100(result_of_mask)
    image_with_mask = img.copy()
    deltas = []
    cnt_tresh120, cnt_tresh90, cnt_tresh75, cnt_tresh60, cnt_tresh45 = 0, 0, 0, 0, 0
    rows, cols = np.where(mask > 0)
    for idx in range(len(rows)):
        j, i = rows[idx], cols[idx]
        color_result1 = image_with_mask[j, i]
        color_result2 = result_of_mask[j, i]
        if color_result2.all() != 0:
            if color_result1[1] == color_result2[1] and (color_result1[0] <= mean_B or color_result1[2] <= mean_R):
                deltaB = (mean_B - color_result1[0]) if color_result1[0] < mean_B else 0
                deltaG = (mean_G - color_result1[1]) if color_result1[1] < mean_G else 0
                deltaR = (mean_R - color_result1[2]) if color_result1[2] < mean_R else 0
                deltaG_plus = (color_result1[1] - mean_G) if color_result1[1] > mean_G else 0
                delta1 = deltaG + deltaR
                if delta1 > 120:
                    color_result1[:] = colors[0]
                    cnt_tresh120 += 1
                elif 120 >= delta1 > 90:
                    color_result1[:] = colors[1]
                    cnt_tresh90 += 1
                elif 90 >= delta1 > 75:
                    color_result1[:] = colors[2]
                    cnt_tresh75 += 1
                elif 75 >= delta1 > 60:
                    color_result1[:] = colors[3]
                    cnt_tresh60 += 1
                elif 60 >= delta1 > 45:
                    color_result1[:] = colors[4]
                    cnt_tresh45 += 1
                if deltaG_plus > 10:
                    color_result1[:] = colors[5]
                deltas.append([deltaB, deltaG, deltaR, delta1, deltaG_plus])
    mean_delta1 = mean_B + mean_G + mean_R
    cnt_tresh25 = 0
    mean_deltaF = 0
    mean_deltaG = 0
    max_deltaF = 0
    max_deltaG = 0
    for delta in deltas:
        if delta[3] > 45:
            cnt_tresh25 += 1
            mean_deltaF += delta[3]
            max_deltaF = max(max_deltaF, delta[3])
        if delta[3] > 120:
            mean_deltaG += delta[3]
            max_deltaG = max(max_deltaG, delta[3])
    if cnt_tresh25 == 0:
        return image_with_mask, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt_all = len(deltas)
    mean_deltaG = round(mean_deltaG / cnt_tresh25, 2)
    changing_the_ratio_of_green_fluorescence = round(mean_deltaG / mean_delta1, 2)
    max_changing_the_ratio_of_green_fluorescence = round(max_deltaG / mean_delta1, 2)
    mean_deltaF = round(mean_deltaF / cnt_tresh25, 2)
    average_fluorescence_loss = round(mean_deltaF / mean_delta1, 2)
    maximum_fluorescence_loss = round(max_deltaF / mean_delta1, 2)
    q0 = cnt_tresh25 / cnt_all
    q1 = cnt_tresh45 / cnt_all
    q2 = cnt_tresh60 / cnt_all
    q3 = cnt_tresh75 / cnt_all
    q4 = cnt_tresh90 / cnt_all
    return image_with_mask, q0, q1, q2, q3, q4, mean_deltaG, \
        changing_the_ratio_of_green_fluorescence, max_changing_the_ratio_of_green_fluorescence, \
        mean_deltaF, average_fluorescence_loss, maximum_fluorescence_loss

        
def add_text_to_image(image, q0, q1, q2, q3, q4, mean_deltaR,
                      changing_the_ratio_of_red_and_green_fluorescence,
                      max_changing_the_ratio_of_red_and_green_fluorescence,
                      mean_deltaF, average_fluorescence_loss,
                      maximum_fluorescence_loss):
    """
    Добавляет текст с результатами анализа на изображение.
    """
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 1
    
    # Текст с результатами
    resultText1 = 'Results:'
    resultText2 = f'mean deltaR: {mean_deltaR:.2f}'
    resultText3 = f'changing the ratio of red and green fluorescence: {changing_the_ratio_of_red_and_green_fluorescence:.2f}'
    resultText4 = f'max changing the ratio of red and green fluorescence: {max_changing_the_ratio_of_red_and_green_fluorescence:.2f}'
    resultText5 = f'mean deltaF: {mean_deltaF:.2f}'
    resultText6 = f'average fluorescence loss: {average_fluorescence_loss:.2f}'
    resultText7 = f'maximum fluorescence loss: {maximum_fluorescence_loss:.2f}'
    resultText8 = f'ALL Q0: {q0*100:.2f}%'
    resultText9 = f' 45 Q1: {q1*100:.2f}%'
    resultText10 = f' 60 Q2: {q2*100:.2f}%'
    resultText11 = f' 75 Q3: {q3*100:.2f}%'
    resultText12 = f' 90 Q4: {q4*100:.2f}%'

    # Получаем размеры изображения
    height, width = image.shape[:2]

    # Вычисляем минимальную высоту для вывода текста
    min_text_height = height * 0.25

    # Если минимальная высота для текста больше текущей высоты изображения, 
    # увеличиваем размеры изображения
    if min_text_height > height:
        scale_factor = min_text_height / height
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        
        
    # Позиция для текста
    y_offset = 50

    # Добавляем текст на изображение
    image = cv2.putText(image, resultText1, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText2, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText3, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText4, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText5, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText6, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText7, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText8, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText9, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText10, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText11, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
    y_offset += 30
    image = cv2.putText(image, resultText12, (200, y_offset), font, fontScale, color, thickness, cv2.LINE_AA)

    return image


def read_json(path_to_json_in):
    with open(path_to_json_in) as json_file:
        data = json.load(json_file)
        
    originalFilePath = data['originalImageUrl']
    points = data['points']
    
    return originalFilePath, points


def make_json(path_to_json_out, path_to_image_out, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        mean_deltaF, \
        average_fluorescence_loss, maximum_fluorescence_loss):
    host = paths.getEnv()["dnsHost"]
    nowString = datetime.datetime.now().replace(microsecond=0).isoformat()
    answer_json = {
        "DateOfAnalisys":nowString,
        "resultImageFilePath":f"{host}/{path_to_image_out}",
        "analisys":{
            "Q0":q0,
            "Q1":q1,
            "Q2":q2,
            "Q3":q3,
            "Q4":q4,
            "mean_deltaR":mean_deltaR,
            "changing_the_ratio_of_red_and_green_fluorescence":changing_the_ratio_of_red_and_green_fluorescence,
            "max_changing_the_ratio_of_red_and_green_fluorescence":max_changing_the_ratio_of_red_and_green_fluorescence,
            "mean_deltaF":mean_deltaF,
            "average_fluorescence_loss":average_fluorescence_loss,
            "maximum_fluorescence_loss":maximum_fluorescence_loss
        }
    }
    
    #answer_json = json.dumps(data)
    with open(path_to_json_out, 'w') as json_file:
        json.dump(answer_json, json_file)

    return answer_json