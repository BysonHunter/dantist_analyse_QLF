from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import base64
import tempfile
import shutil
from datetime import datetime
import sys
from pathlib import Path
import warnings
import zipfile
from io import BytesIO
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Импортируем ваши модули
try:
    import dantistUtils as du
    import colorpalette as cp
    colors = cp.firecolorpalette
    colors_rgb = [(c[2], c[1], c[0]) for c in colors]  # Конвертация BGR в RGB
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    colors = [
        (0, 255, 255),  # yellow
        (0, 215, 255),  # gold
        (0, 165, 155),  # orange
        (0, 140, 255),  # darkorange
        (0, 69, 255),   # orangered
        (0, 0, 255),    # red
    ]
    colors_rgb = [(c[2], c[1], c[0]) for c in colors]

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Увеличиваем максимальный размер запроса, чтобы спокойно передавать изображения в base64
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['RESULTS_FOLDER'] = 'results'

# Максимальный размер стороны изображения при загрузке (с сохранением пропорций)
MAX_IMAGE_DIMENSION = 1920

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class QLFAnalyzer:
    def __init__(self):
        self.image = None
        self.results = {}
        
    def load_image_from_base64(self, base64_string):
        """Загрузка изображения из base64 строки"""
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.image
    
    def load_image_from_file(self, file_path):
        """Загрузка изображения из файла"""
        self.image = cv2.imread(file_path)
        return self.image
    
    def analyze_roi_with_segmentation(self, roi_coords):
        """
        Анализ выбранной области ROI с предварительной сегментацией зубов
        Новый алгоритм: ROI -> сегментация зубов в ROI -> анализ отсегментированных областей
        """
        if self.image is None or roi_coords is None:
            return None, None
        
        x, y, w, h = roi_coords
        
        # Вырезаем ROI из изображения
        roi_image = self.image[y:y+h, x:x+w].copy()
        
        if roi_image.size == 0:
            return None, None
        
        # Находим контуры зубов в ROI
        _, _, contours_roi, _ = du.find_conturs(roi_image)
        
        # Преобразуем контуры в глобальные координаты
        contours_global = []
        if contours_roi:
            for contour in contours_roi:
                # Добавляем смещение ROI к контуру
                contour_global = contour + np.array([x, y])
                contours_global.append(contour_global)
        
        if contours_global:
            result = du.tooth_Analise(self.image, contours_global, colors)
            result_green = du.tooth_Analise_green(self.image, contours_global, colors)
            contours_for_qlf = contours_global
        else:
            roi_contour = np.array([[
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]], dtype=np.int32)
            result = du.select_tooth_Analise(self.image, roi_contour, colors)
            result_green = du.select_tooth_Analise_green(self.image, roi_contour, colors)
            contours_for_qlf = roi_contour
        
        if len(result) >= 12 and len(result_green) >= 12:
            (image_with_mask, q0, q1, q2, q3, q4,
             mean_deltaR, changing_ratio, max_changing_ratio,
             mean_deltaF, avg_fluorescence_loss, max_fluorescence_loss) = result
            (image_green, q0g, q1g, q2g, q3g, q4g,
             mean_deltaG, changing_g, max_changing_g,
             mean_deltaF_g, avg_fluor_g, max_fluor_g) = result_green

            roi_stats = {
                'q0': float(q0), 'q1': float(q1), 'q2': float(q2), 'q3': float(q3), 'q4': float(q4),
                'mean_deltaR': float(mean_deltaR),
                'changing_the_ratio_of_red_and_green_fluorescence': float(changing_ratio),
                'max_changing_the_ratio_of_red_and_green_fluorescence': float(max_changing_ratio),
                'mean_deltaF': float(mean_deltaF),
                'average_fluorescence_loss': float(avg_fluorescence_loss),
                'maximum_fluorescence_loss': float(max_fluorescence_loss),
                'coords': roi_coords,
                'contours_found': len(contours_global) > 0
            }
            roi_stats_green = {
                'q0': float(q0g), 'q1': float(q1g), 'q2': float(q2g), 'q3': float(q3g), 'q4': float(q4g),
                'mean_deltaG': float(mean_deltaG),
                'changing_the_ratio_of_green_fluorescence': float(changing_g),
                'max_changing_the_ratio_of_green_fluorescence': float(max_changing_g),
                'mean_deltaF': float(mean_deltaF_g),
                'average_fluorescence_loss': float(avg_fluor_g),
                'maximum_fluorescence_loss': float(max_fluor_g),
                'coords': roi_coords,
                'contours_found': len(contours_global) > 0
            }
            self.results['roi'] = {
                'stats': roi_stats,
                'visualization': image_with_mask,
                'contours_count': len(contours_global)
            }
            self.results['roi_green'] = {
                'stats': roi_stats_green,
                'visualization': image_green,
            }
            roi_qlf_stats, roi_qlf_vis = self._analyze_qlf_df(contours_for_qlf)
            self.results['roi_qlf_df'] = {
                'stats': roi_qlf_stats,
                'visualization': roi_qlf_vis,
            }
            return roi_stats, image_with_mask
        
        return None, None
    
    def analyze_all_teeth(self):
        """Анализ всех зубов с автоматической сегментацией"""
        if self.image is None:
            return None, None
        
        # Находим контуры зубов
        img_with_conturs, mask, contours, hierarchy = du.find_conturs(self.image)
        
        if not contours:
            print("Не удалось найти контуры зубов")
            return None, None
        
        result = du.tooth_Analise(self.image, contours, colors)
        result_green = du.tooth_Analise_green(self.image, contours, colors)

        if len(result) >= 12 and len(result_green) >= 12:
            (image_with_mask, q0, q1, q2, q3, q4,
             mean_deltaR, changing_ratio, max_changing_ratio,
             mean_deltaF, avg_fluorescence_loss, max_fluorescence_loss) = result
            (image_green, q0g, q1g, q2g, q3g, q4g,
             mean_deltaG, changing_g, max_changing_g,
             mean_deltaF_g, avg_fluor_g, max_fluor_g) = result_green

            all_stats = {
                'q0': float(q0), 'q1': float(q1), 'q2': float(q2), 'q3': float(q3), 'q4': float(q4),
                'mean_deltaR': float(mean_deltaR),
                'changing_the_ratio_of_red_and_green_fluorescence': float(changing_ratio),
                'max_changing_the_ratio_of_red_and_green_fluorescence': float(max_changing_ratio),
                'mean_deltaF': float(mean_deltaF),
                'average_fluorescence_loss': float(avg_fluorescence_loss),
                'maximum_fluorescence_loss': float(max_fluorescence_loss),
                'total_contours': len(contours)
            }
            all_stats_green = {
                'q0': float(q0g), 'q1': float(q1g), 'q2': float(q2g), 'q3': float(q3g), 'q4': float(q4g),
                'mean_deltaG': float(mean_deltaG),
                'changing_the_ratio_of_green_fluorescence': float(changing_g),
                'max_changing_the_ratio_of_green_fluorescence': float(max_changing_g),
                'mean_deltaF': float(mean_deltaF_g),
                'average_fluorescence_loss': float(avg_fluor_g),
                'maximum_fluorescence_loss': float(max_fluor_g),
                'total_contours': len(contours)
            }
            self.results['all_teeth'] = {
                'stats': all_stats,
                'visualization': image_with_mask,
                'contours': contours,
                'hierarchy': hierarchy
            }
            self.results['all_teeth_green'] = {
                'stats': all_stats_green,
                'visualization': image_green,
            }
            all_qlf_stats, all_qlf_vis = self._analyze_qlf_df(contours)
            self.results['all_teeth_qlf_df'] = {
                'stats': all_qlf_stats,
                'visualization': all_qlf_vis,
            }
            return all_stats, image_with_mask

        return None, None

    def _analyze_qlf_df(self, contours):
        """Третий анализ: QLF DF по 3-й ячейке new_analyse2.ipynb."""
        empty_stats = {
            'total_lesions': 0,
            'mean_df_percent': 0.0,
            'total_lesion_area_px': 0,
            'total_dq': 0.0,
            'healthy_area_px': 0,
            'enamel_affected_percent': 0.0,
            'q10': 0.0,
            'q45': 0.0,
            'q60': 0.0,
            'q75': 0.0,
            'q90': 0.0,
        }
        if self.image is None or contours is None or len(contours) == 0:
            return empty_stats, None

        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        green_channel = self.image[:, :, 1].astype(np.float32)
        tooth_pixels = green_channel[mask > 0]
        if tooth_pixels.size == 0:
            return empty_stats, None

        max_fl_intensity = float(np.percentile(tooth_pixels, 95))
        if max_fl_intensity < 1e-3:
            max_fl_intensity = 1.0
        delta_f_map = np.zeros(self.image.shape[:2], dtype=np.float32)
        delta_f_map[mask > 0] = (1.0 - green_channel[mask > 0] / max_fl_intensity) * 100.0
        delta_f_map = np.clip(delta_f_map, 0.0, 100.0)

        lesion_mask = ((delta_f_map > 10) & (mask > 0)).astype(np.uint8)
        lesion_mask = self._remove_small_components(lesion_mask, min_size=20)

        lesion_pixels = int(np.sum(lesion_mask > 0))
        tooth_total = int(np.sum(mask > 0))
        healthy_pixels = max(0, tooth_total - lesion_pixels)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_mask, connectivity=8)
        lesion_count = 0
        lesion_dq_sum = 0.0
        lesion_df_values = []
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            lesion_count += 1
            comp_mask = labels == idx
            mean_df = float(np.mean(delta_f_map[comp_mask]))
            lesion_df_values.append(mean_df)
            lesion_dq_sum += float(mean_df * area)

        q10 = float(np.sum((delta_f_map > 10) & (mask > 0)) / tooth_total) if tooth_total else 0.0
        q45 = float(np.sum((delta_f_map > 45) & (mask > 0)) / tooth_total) if tooth_total else 0.0
        q60 = float(np.sum((delta_f_map > 60) & (mask > 0)) / tooth_total) if tooth_total else 0.0
        q75 = float(np.sum((delta_f_map > 75) & (mask > 0)) / tooth_total) if tooth_total else 0.0
        q90 = float(np.sum((delta_f_map > 90) & (mask > 0)) / tooth_total) if tooth_total else 0.0

        stats_out = {
            'total_lesions': int(lesion_count),
            'mean_df_percent': float(np.mean(lesion_df_values)) if lesion_df_values else 0.0,
            'total_lesion_area_px': int(lesion_pixels),
            'total_dq': float(lesion_dq_sum),
            'healthy_area_px': int(healthy_pixels),
            'enamel_affected_percent': float((lesion_pixels / tooth_total) * 100.0) if tooth_total else 0.0,
            'q10': q10,
            'q45': q45,
            'q60': q60,
            'q75': q75,
            'q90': q90,
        }

        vis = self._build_qlf_df_visualization(delta_f_map, lesion_mask, mask)
        return stats_out, vis

    def _remove_small_components(self, binary_mask, min_size=20):
        """Удаляет мелкие связные компоненты без scipy/skimage."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_size:
                cleaned[labels == idx] = 1
        return cleaned

    def _build_qlf_df_visualization(self, delta_f_map, lesion_mask, mask):
        """Визуализация QLF DF: наложение fire palette на исходное изображение."""
        vis = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB).astype(np.float32)
        overlay = np.zeros_like(vis, dtype=np.float32)
        overlay[(delta_f_map >= 10) & (delta_f_map < 45) & (lesion_mask == 1) & (mask > 0)] = tuple(reversed(colors[4]))
        overlay[(delta_f_map >= 45) & (delta_f_map < 60) & (lesion_mask == 1) & (mask > 0)] = tuple(reversed(colors[3]))
        overlay[(delta_f_map >= 60) & (delta_f_map < 75) & (lesion_mask == 1) & (mask > 0)] = tuple(reversed(colors[2]))
        overlay[(delta_f_map >= 75) & (delta_f_map < 90) & (lesion_mask == 1) & (mask > 0)] = tuple(reversed(colors[1]))
        overlay[(delta_f_map >= 90) & (lesion_mask == 1) & (mask > 0)] = tuple(reversed(colors[0]))
        alpha = 0.65
        lesion_pixels = (lesion_mask == 1) & (mask > 0)
        vis[lesion_pixels] = (1.0 - alpha) * vis[lesion_pixels] + alpha * overlay[lesion_pixels]
        return cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)

def image_to_base64(image, convert_to_rgb=True):
    """Конвертация изображения в base64 строку"""
    if image is None:
        return None
    
    # Создаем копию изображения, чтобы не изменять оригинал
    img_copy = image.copy()
    
    # Конвертируем BGR в RGB если нужно
    # Предполагаем, что на вход приходит изображение в формате OpenCV (BGR),
    # поэтому при convert_to_rgb всегда выполняем cvtColor без эвристик по пикселям
    if convert_to_rgb and len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    
    _, buffer = cv2.imencode('.png', img_copy)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def save_image_to_file(image, filepath, convert_to_rgb=True):
    """Сохранение изображения в файл с правильными цветами"""
    if image is None:
        return False
    
    # Создаем копию изображения, чтобы не изменять оригинал
    img_copy = image.copy()
    
    # Конвертируем BGR в RGB если нужно
    if convert_to_rgb and len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
        if img_copy[0,0,0] != img_copy[0,0,2]:  # Если B и R разные, значит это BGR
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    
    # Сохраняем изображение
    return cv2.imwrite(filepath, img_copy)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Загрузка изображения на сервер"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Сохраняем файл во временное место и загружаем
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    height, width = image.shape[:2]
    
    # Уменьшаем изображение с сохранением пропорций, если больше MAX_IMAGE_DIMENSION
    if max(width, height) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(width, height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filepath, image)
        height, width = image.shape[:2]
    
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath,
        'width': width,
        'height': height,
        'url': f'/uploads/{filename}'
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Анализ изображения с ROI"""
    data = request.json
    filename = data.get('filename')
    roi_coords = data.get('roi_coords')
    
    if not filename or not roi_coords:
        return jsonify({'error': 'Missing filename or ROI coordinates'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Создаем анализатор
    analyzer = QLFAnalyzer()
    analyzer.load_image_from_file(filepath)
    
    # Анализ ROI с новой логикой сегментации
    roi_stats, _ = analyzer.analyze_roi_with_segmentation(roi_coords)
    
    # Анализ всех зубов
    all_stats, _ = analyzer.analyze_all_teeth()
    
    if roi_stats is None or all_stats is None:
        return jsonify({'error': 'Analysis failed'}), 500
    
    roi_image_bgr = analyzer.results['roi']['visualization']
    all_image_bgr = analyzer.results['all_teeth']['visualization']
    roi_green_bgr = analyzer.results['roi_green']['visualization']
    all_green_bgr = analyzer.results['all_teeth_green']['visualization']
    roi_qlf_df_bgr = analyzer.results['roi_qlf_df']['visualization']
    all_qlf_df_bgr = analyzer.results['all_teeth_qlf_df']['visualization']
    roi_stats_green = analyzer.results['roi_green']['stats']
    all_stats_green = analyzer.results['all_teeth_green']['stats']
    roi_stats_qlf_df = analyzer.results['roi_qlf_df']['stats']
    all_stats_qlf_df = analyzer.results['all_teeth_qlf_df']['stats']

    roi_image_base64 = image_to_base64(roi_image_bgr, convert_to_rgb=False)
    all_image_base64 = image_to_base64(all_image_bgr, convert_to_rgb=False)
    roi_green_base64 = image_to_base64(roi_green_bgr, convert_to_rgb=False)
    all_green_base64 = image_to_base64(all_green_bgr, convert_to_rgb=False)
    roi_qlf_df_base64 = image_to_base64(roi_qlf_df_bgr, convert_to_rgb=False)
    all_qlf_df_base64 = image_to_base64(all_qlf_df_bgr, convert_to_rgb=False)

    # Цветовая легенда для Red/Green delta-анализа
    color_legend = []
    color_legend.append({
        'color': 'rgb(200, 200, 200)',
        'label': 'Healthy / Delta < 45'
    })
    labels = [
        "Delta > 120 (Серьезное поражение)",
        "90 < Delta ≤ 120",
        "75 < Delta ≤ 90",
        "60 < Delta ≤ 75",
        "45 < Delta ≤ 60",
        "Красная флуоресценция (DeltaR+)"
    ]
    
    for i, (color, label) in enumerate(zip(colors_rgb, labels)):
        color_legend.append({
            'color': f'rgb({color[0]}, {color[1]}, {color[2]})',
            'label': label
        })

    # Отдельная легенда для QLF DF (алгоритм из 3-й ячейки new_analyse2.ipynb)
    color_legend_qlf_df = [
        {'color': 'rgb(160, 160, 160)', 'label': 'DF < 10% (Healthy)'},
        {'color': f'rgb({colors_rgb[4][0]}, {colors_rgb[4][1]}, {colors_rgb[4][2]})', 'label': 'DF 10-45% (Mild)'},
        {'color': f'rgb({colors_rgb[3][0]}, {colors_rgb[3][1]}, {colors_rgb[3][2]})', 'label': 'DF 45-60% (Moderate)'},
        {'color': f'rgb({colors_rgb[2][0]}, {colors_rgb[2][1]}, {colors_rgb[2][2]})', 'label': 'DF 60-75% (Severe)'},
        {'color': f'rgb({colors_rgb[1][0]}, {colors_rgb[1][1]}, {colors_rgb[1][2]})', 'label': 'DF 75-90% (Critical)'},
        {'color': f'rgb({colors_rgb[0][0]}, {colors_rgb[0][1]}, {colors_rgb[0][2]})', 'label': 'DF 90-100% (Extreme)'},
    ]
    
    return jsonify({
        'success': True,
        'roi_analysis': {
            'stats': roi_stats,
            'image': roi_image_base64,
            'contours_found': roi_stats['contours_found']
        },
        'all_teeth_analysis': {
            'stats': all_stats,
            'image': all_image_base64
        },
        'roi_analysis_green': {
            'stats': roi_stats_green,
            'image': roi_green_base64,
        },
        'all_teeth_analysis_green': {
            'stats': all_stats_green,
            'image': all_green_base64
        },
        'roi_analysis_qlf_df': {
            'stats': roi_stats_qlf_df,
            'image': roi_qlf_df_base64,
        },
        'all_teeth_analysis_qlf_df': {
            'stats': all_stats_qlf_df,
            'image': all_qlf_df_base64
        },
        'color_legend': color_legend,
        'color_legend_qlf_df': color_legend_qlf_df,
        'analyzer_data': {
            'roi_image': roi_image_base64,
            'all_teeth_image': all_image_base64,
            'roi_image_green': roi_green_base64,
            'all_teeth_image_green': all_green_base64,
            'roi_image_qlf_df': roi_qlf_df_base64,
            'all_teeth_image_qlf_df': all_qlf_df_base64
        }
    })

@app.route('/save_results', methods=['POST'])
def save_results():
    """Сохранение результатов анализа"""
    data = request.json
    filename = data.get('filename')
    roi_stats = data.get('roi_stats')
    all_stats = data.get('all_stats')
    roi_stats_green = data.get('roi_stats_green')
    all_stats_green = data.get('all_stats_green')
    roi_stats_qlf_df = data.get('roi_stats_qlf_df')
    all_stats_qlf_df = data.get('all_stats_qlf_df')
    roi_image_base64 = data.get('roi_image')
    all_image_base64 = data.get('all_teeth_image')
    roi_image_green_base64 = data.get('roi_image_green')
    all_image_green_base64 = data.get('all_teeth_image_green')
    roi_image_qlf_df_base64 = data.get('roi_image_qlf_df')
    all_image_qlf_df_base64 = data.get('all_teeth_image_qlf_df')
    
    if not filename or not roi_stats or not roi_image_base64 or not all_image_base64:
        return jsonify({'error': 'Missing data for saving'}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"qlf_analysis_{timestamp}"
    
    try:
        # Декодируем изображения из base64
        roi_b64 = roi_image_base64.split(',')[1] if ',' in roi_image_base64 else roi_image_base64
        all_b64 = all_image_base64.split(',')[1] if ',' in all_image_base64 else all_image_base64
        roi_image_data = base64.b64decode(roi_b64)
        all_image_data = base64.b64decode(all_b64)
        roi_nparr = np.frombuffer(roi_image_data, np.uint8)
        all_nparr = np.frombuffer(all_image_data, np.uint8)
        roi_img = cv2.imdecode(roi_nparr, cv2.IMREAD_COLOR)
        all_img = cv2.imdecode(all_nparr, cv2.IMREAD_COLOR)
        if roi_img is None or all_img is None:
            raise ValueError("Failed to decode images from base64")

        # Кодируем PNG в байты (без записи на диск)
        _, roi_png = cv2.imencode('.png', roi_img)
        _, all_png = cv2.imencode('.png', all_img)
        roi_png_bytes = roi_png.tobytes()
        all_png_bytes = all_png.tobytes()

        report_content = create_report(filename, roi_stats, all_stats)
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'original_filename': filename,
            'roi_analysis': roi_stats,
            'all_teeth_analysis': all_stats if all_stats else None
        }
        json_bytes = json.dumps(json_data, indent=4, ensure_ascii=False).encode('utf-8')

        # Собираем ZIP в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(f"{base_name}_roi.png", roi_png_bytes)
            zipf.writestr(f"{base_name}_all_teeth.png", all_png_bytes)
            zipf.writestr(f"{base_name}_report.txt", report_content.encode('utf-8'))
            zipf.writestr(f"{base_name}_data.json", json_bytes)

            if roi_image_green_base64 and all_image_green_base64 and roi_stats_green and all_stats_green:
                r_g = roi_image_green_base64.split(',')[1] if ',' in roi_image_green_base64 else roi_image_green_base64
                a_g = all_image_green_base64.split(',')[1] if ',' in all_image_green_base64 else all_image_green_base64
                roi_green_arr = np.frombuffer(base64.b64decode(r_g), np.uint8)
                all_green_arr = np.frombuffer(base64.b64decode(a_g), np.uint8)
                roi_green_img = cv2.imdecode(roi_green_arr, cv2.IMREAD_COLOR)
                all_green_img = cv2.imdecode(all_green_arr, cv2.IMREAD_COLOR)
                if roi_green_img is not None and all_green_img is not None:
                    _, roi_green_png = cv2.imencode('.png', roi_green_img)
                    _, all_green_png = cv2.imencode('.png', all_green_img)
                    zipf.writestr(f"{base_name}_roi_green.png", roi_green_png.tobytes())
                    zipf.writestr(f"{base_name}_all_teeth_green.png", all_green_png.tobytes())
                json_green_data = {
                    'timestamp': datetime.now().isoformat(),
                    'original_filename': filename,
                    'channel': 'green',
                    'roi_analysis': roi_stats_green,
                    'all_teeth_analysis': all_stats_green
                }
                zipf.writestr(f"{base_name}_green_data.json",
                    json.dumps(json_green_data, indent=4, ensure_ascii=False).encode('utf-8'))

            if roi_image_qlf_df_base64 and all_image_qlf_df_base64 and roi_stats_qlf_df and all_stats_qlf_df:
                r_q = roi_image_qlf_df_base64.split(',')[1] if ',' in roi_image_qlf_df_base64 else roi_image_qlf_df_base64
                a_q = all_image_qlf_df_base64.split(',')[1] if ',' in all_image_qlf_df_base64 else all_image_qlf_df_base64
                roi_qlf_arr = np.frombuffer(base64.b64decode(r_q), np.uint8)
                all_qlf_arr = np.frombuffer(base64.b64decode(a_q), np.uint8)
                roi_qlf_img = cv2.imdecode(roi_qlf_arr, cv2.IMREAD_COLOR)
                all_qlf_img = cv2.imdecode(all_qlf_arr, cv2.IMREAD_COLOR)
                if roi_qlf_img is not None and all_qlf_img is not None:
                    _, roi_qlf_png = cv2.imencode('.png', roi_qlf_img)
                    _, all_qlf_png = cv2.imencode('.png', all_qlf_img)
                    zipf.writestr(f"{base_name}_roi_qlf_df.png", roi_qlf_png.tobytes())
                    zipf.writestr(f"{base_name}_all_teeth_qlf_df.png", all_qlf_png.tobytes())
                json_qlf_data = {
                    'timestamp': datetime.now().isoformat(),
                    'original_filename': filename,
                    'channel': 'qlf_df_green',
                    'roi_analysis': roi_stats_qlf_df,
                    'all_teeth_analysis': all_stats_qlf_df
                }
                zipf.writestr(
                    f"{base_name}_qlf_df_data.json",
                    json.dumps(json_qlf_data, indent=4, ensure_ascii=False).encode('utf-8')
                )

        # Записываем готовый ZIP на диск одним вызовом
        zip_filename = f"{base_name}.zip"
        zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)
        with open(zip_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        # Возвращаем информацию для скачивания
        return jsonify({
            'success': True,
            'message': 'Результаты успешно сохранены',
            'zip_filename': zip_filename,
            'download_url': f'/download/{zip_filename}'
        })
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return jsonify({'error': f'Failed to save results: {str(e)}'}), 500

def create_report(filename, roi_stats, all_stats):
    """Создание текстового отчета"""
    report = f"""QLF АНАЛИЗ ЗУБНОЙ ЭМАЛИ
Отчет сгенерирован: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Файл изображения: {filename}
Используемая цветовая палитра: firecolorpalette
Алгоритм анализа ROI: Сегментация зубов в области ROI -> Анализ отсегментированных областей

1. АНАЛИЗ ВЫБРАННОЙ ОБЛАСТИ (ROI):
   Координаты: X={roi_stats.get('coords', [0,0,0,0])[0]}, Y={roi_stats.get('coords', [0,0,0,0])[1]}, 
   Ширина={roi_stats.get('coords', [0,0,0,0])[2]}, Высота={roi_stats.get('coords', [0,0,0,0])[3]}
   Контуры зубов найдены в ROI: {'Да' if roi_stats.get('contours_found', False) else 'Нет'}
   
   Распределение поражений по уровням Delta:
   - Q0 (Все поражения Delta > 45): {roi_stats.get('q0', 0)*100:.1f}%
   - Q1 (45 < Delta ≤ 60): {roi_stats.get('q1', 0)*100:.1f}%
   - Q2 (60 < Delta ≤ 75): {roi_stats.get('q2', 0)*100:.1f}%
   - Q3 (75 < Delta ≤ 90): {roi_stats.get('q3', 0)*100:.1f}%
   - Q4 (Delta > 90): {roi_stats.get('q4', 0)*100:.1f}%
   
   Метрики QLF:
   - mean_deltaR: {roi_stats.get('mean_deltaR', 0):.2f}
   - mean_deltaF: {roi_stats.get('mean_deltaF', 0):.2f}
   - changing_the_ratio_of_red_and_green_fluorescence: {roi_stats.get('changing_the_ratio_of_red_and_green_fluorescence', 0):.2f}
   - max_changing_the_ratio_of_red_and_green_fluorescence: {roi_stats.get('max_changing_the_ratio_of_red_and_green_fluorescence', 0):.2f}
   - average_fluorescence_loss: {roi_stats.get('average_fluorescence_loss', 0):.2f}
   - maximum_fluorescence_loss: {roi_stats.get('maximum_fluorescence_loss', 0):.2f}

"""
    
    if all_stats:
        report += f"""
2. АНАЛИЗ ВСЕХ ЗУБОВ (АВТОМАТИЧЕСКАЯ СЕГМЕНТАЦИЯ):
   Всего обнаружено контуров зубов: {all_stats.get('total_contours', 0)}
   
   Распределение поражений по уровням Delta:
   - Q0 (Все поражения Delta > 45): {all_stats.get('q0', 0)*100:.1f}%
   - Q1 (45 < Delta ≤ 60): {all_stats.get('q1', 0)*100:.1f}%
   - Q2 (60 < Delta ≤ 75): {all_stats.get('q2', 0)*100:.1f}%
   - Q3 (75 < Delta ≤ 90): {all_stats.get('q3', 0)*100:.1f}%
   - Q4 (Delta > 90): {all_stats.get('q4', 0)*100:.1f}%
   
   Метрики QLF:
   - mean_deltaR: {all_stats.get('mean_deltaR', 0):.2f}
   - mean_deltaF: {all_stats.get('mean_deltaF', 0):.2f}
   - changing_the_ratio_of_red_and_green_fluorescence: {all_stats.get('changing_the_ratio_of_red_and_green_fluorescence', 0):.2f}
   - max_changing_the_ratio_of_red_and_green_fluorescence: {all_stats.get('max_changing_the_ratio_of_red_and_green_fluorescence', 0):.2f}
   - average_fluorescence_loss: {all_stats.get('average_fluorescence_loss', 0):.2f}
   - maximum_fluorescence_loss: {all_stats.get('maximum_fluorescence_loss', 0):.2f}
"""
    
    return report

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание файла"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Очистка старых файлов"""
    try:
        # Удаляем файлы старше 24 часов
        current_time = datetime.now().timestamp()
        max_age = 24 * 3600  # 24 часа в секундах
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            if not os.path.exists(folder):
                continue
                
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age:
                        os.remove(filepath)
                        print(f"Удален старый файл: {filepath}")
        
        return jsonify({'success': True, 'message': 'Очистка завершена'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)