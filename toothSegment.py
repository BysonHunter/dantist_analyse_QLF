import cv2 
import numpy as np
import requests
import paths
from os.path import join

import dantistUtils as du
import colorpalette as cp

colors = cp.graycolorpalette
#colors = cp.firecolorpalette

def main(image_file_name, points, json_path):
    """
    Основная функция для анализа изображения.
    """

    img_origin = cv2.imread(image_file_name)
    if img_origin is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_file_name}'")
        return

    # Анализ изображения
    img_with_conturs, mask, contours, hierarchy = du.find_conturs(img_origin)
    cv2.drawContours(img_with_conturs, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 1)

    # Анализ всех зубов с использованием полученных контуров
    image_with_mask, q0, q1, q2, q3, q4, mean_deltaR, \
    changing_the_ratio_of_red_and_green_fluorescence, \
    max_changing_the_ratio_of_red_and_green_fluorescence, \
    mean_deltaF, \
    average_fluorescence_loss, maximum_fluorescence_loss = du.tooth_Analise(img_origin, contours, colors)
    
    image_with_text = image_with_mask
    
    # Сохранение изображения
    filename_image_to_save = f'{image_file_name[:-4]}_result.png'
    cv2.imwrite(filename=filename_image_to_save, img=image_with_text)
    
    return filename_image_to_save, f'{q0*100:.2f}%', f'{q1*100:.2f}%', f'{q2*100:.2f}%', f'{q3*100:.2f}%', f'{q4*100:.2f}%' , \
            mean_deltaR, \
            changing_the_ratio_of_red_and_green_fluorescence, \
            max_changing_the_ratio_of_red_and_green_fluorescence, \
            mean_deltaF, \
            average_fluorescence_loss, maximum_fluorescence_loss
    

def processImage(image_file, points, path_to_json_out):
    image_result_filename, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, mean_deltaF, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        average_fluorescence_loss, maximum_fluorescence_loss = main(image_file, points, path_to_json_out)

    return du.make_json(path_to_json_out, image_result_filename, q0, q1, q2, q3, q4, mean_deltaR, \
              changing_the_ratio_of_red_and_green_fluorescence, \
              max_changing_the_ratio_of_red_and_green_fluorescence, \
              mean_deltaF, \
              average_fluorescence_loss, maximum_fluorescence_loss)


if __name__ == "__main__":
    path_to_json_in = join(paths.rootFolder,'dantist_in.json')
    path_to_json_out = join(paths.rootFolder,'dantist_out.json')
    path_to_image_file = join(paths.rootFolder,'input_image.jpg')

    try:
        file = open(path_to_json_in)
    except IOError as e:
        print(u'не удалось открыть json файл')

    else:
        image_url, points = du.read_json(path_to_json_in)
        img_data = requests.get(image_url).content
        image_file = open(path_to_image_file, 'wb')
        image_file.write(img_data)
        image_file.close()
        processImage(path_to_image_file, points, path_to_json_out)