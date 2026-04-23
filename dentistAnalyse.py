import cv2 
import numpy as np 
import json
import paths

import requests
import datetime
from os.path import join

import dantistUtils as du
import colorpalette as cp

colors = cp.graycolorpalette
#colors = cp.firecolorpalette


def main(imgfile, points):
    img = cv2.imread(imgfile)
    originframeSize = (img.shape[1],img.shape[0])
    points_to_draw = np.array([points])
    image_with_mask, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        mean_deltaF, \
        average_fluorescence_loss, maximum_fluorescence_loss = du.select_tooth_Analise(img, points_to_draw, colors)
    
    filename=f'{imgfile[:-4]}_result.jpg'
    cv2.imwrite(filename=filename, img=cv2.resize(image_with_mask, originframeSize))

    return filename, f'{q0*100:.2f}%', f'{q1*100:.2f}%', f'{q2*100:.2f}%', f'{q3*100:.2f}%', f'{q4*100:.2f}%' , \
        mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        mean_deltaF, \
        average_fluorescence_loss, maximum_fluorescence_loss


def processImage(image_file, points, path_to_json_out):
    image_result_filename, q0, q1, q2, q3, q4, mean_deltaR, \
        changing_the_ratio_of_red_and_green_fluorescence, mean_deltaF, \
        max_changing_the_ratio_of_red_and_green_fluorescence, \
        average_fluorescence_loss, maximum_fluorescence_loss = main(image_file, points)

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


