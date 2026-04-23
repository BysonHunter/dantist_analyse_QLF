import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import tempfile
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Импортируем ваши модули
try:
    import dantistUtils as du
    import colorpalette as cp
    colors = cp.firecolorpalette
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    colors = [
        (0, 255, 255),  # yellow
        (0, 215, 255),  # gold
        (0, 165, 155),  # orange
        (0, 140, 255),  # darkorange
        (0, 69, 255),   # orangered
        (0, 0, 255),    # red
    ]

# Настройка страницы
st.set_page_config(
    page_title="QLF Анализатор зубной эмали",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для стилизации
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B6CA8;
    }
    .roi-control-group {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .roi-control-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 8px 0;
    }
    .roi-label {
        font-weight: bold;
        width: 80px;
    }
    .roi-controls {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .roi-input {
        width: 80px;
        text-align: center;
        padding: 4px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
    .roi-btn {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 1px solid #2E86AB;
        background-color: white;
        color: #2E86AB;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .roi-btn:hover {
        background-color: #f0f8ff;
    }
    .image-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        margin: 10px 0;
        padding: 5px;
    }
    .reset-btn {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        cursor: pointer;
    }
    .reset-btn:hover {
        background-color: #5a6268;
    }
</style>
""", unsafe_allow_html=True)

class QLFAnalyzer:
    def __init__(self):
        self.image = None
        self.roi_coords = None
        self.segmented_teeth = []
        self.results = {}
        
    def load_image(self, uploaded_file):
        """Загрузка изображения из Streamlit"""
        image = Image.open(uploaded_file)
        self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.image
    
    def analyze_roi(self, roi_coords):
        """Анализ выбранной области ROI"""
        if self.image is None or roi_coords is None:
            return None, None
        
        x, y, w, h = roi_coords
        
        # Создаем прямоугольный контур для ROI
        roi_contour = np.array([[
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]], dtype=np.int32)
        
        # Анализ ROI области
        result = du.select_tooth_Analise(self.image, roi_contour, colors)
        
        if len(result) >= 12:
            (image_with_mask, q0, q1, q2, q3, q4, 
             mean_deltaR, changing_ratio, max_changing_ratio,
             mean_deltaF, avg_fluorescence_loss, max_fluorescence_loss) = result
            
            roi_stats = {
                'q0': float(q0),
                'q1': float(q1),
                'q2': float(q2),
                'q3': float(q3),
                'q4': float(q4),
                'mean_deltaR': float(mean_deltaR),
                'changing_the_ratio_of_red_and_green_fluorescence': float(changing_ratio),
                'max_changing_the_ratio_of_red_and_green_fluorescence': float(max_changing_ratio),
                'mean_deltaF': float(mean_deltaF),
                'average_fluorescence_loss': float(avg_fluorescence_loss),
                'maximum_fluorescence_loss': float(max_fluorescence_loss),
                'coords': roi_coords
            }
            
            self.results['roi'] = {
                'stats': roi_stats,
                'visualization': image_with_mask,
                'roi_colored': image_with_mask[y:y+h, x:x+w] if h > 0 and w > 0 else image_with_mask
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
            st.warning("Не удалось найти контуры зубов")
            return None, None
        
        # Анализ всех зубов
        result = du.tooth_Analise(self.image, contours, colors)
        
        if len(result) >= 12:
            (image_with_mask, q0, q1, q2, q3, q4, 
             mean_deltaR, changing_ratio, max_changing_ratio,
             mean_deltaF, avg_fluorescence_loss, max_fluorescence_loss) = result
            
            # Собираем статистику по всем зубам
            all_stats = {
                'q0': float(q0),
                'q1': float(q1),
                'q2': float(q2),
                'q3': float(q3),
                'q4': float(q4),
                'mean_deltaR': float(mean_deltaR),
                'changing_the_ratio_of_red_and_green_fluorescence': float(changing_ratio),
                'max_changing_the_ratio_of_red_and_green_fluorescence': float(max_changing_ratio),
                'mean_deltaF': float(mean_deltaF),
                'average_fluorescence_loss': float(avg_fluorescence_loss),
                'maximum_fluorescence_loss': float(max_fluorescence_loss),
                'total_contours': len(contours)
            }
            
            self.results['all_teeth'] = {
                'stats': all_stats,
                'visualization': image_with_mask,
                'contours': contours,
                'hierarchy': hierarchy
            }
            
            return all_stats, image_with_mask
        
        return None, None
    
    def convert_bgr_to_rgb(self, image):
        """Конвертация из BGR в RGB для отображения в Streamlit"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def create_fire_legend():
    """Создание легенды цветовой схемы"""
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Цвета из firecolorpalette (BGR -> RGB)
    colors_bgr = [
        (0, 255, 255),  # yellow
        (0, 215, 255),  # gold
        (0, 165, 155),  # orange
        (0, 140, 255),  # darkorange
        (0, 69, 255),   # orangered
        (0, 0, 255),    # red
    ]
    
    colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_bgr]
    
    labels = [
        "Delta > 120 (Серьезное поражение)",
        "90 < Delta ≤ 120",
        "75 < Delta ≤ 90",
        "60 < Delta ≤ 75",
        "45 < Delta ≤ 60",
        "Красная флуоресценция (DeltaR+)"
    ]
    
    for i, (color, label) in enumerate(zip(colors_rgb, labels)):
        ax.barh(0, 1, left=i, color=color, edgecolor='black', height=0.6)
        ax.text(i + 0.5, 0, label, ha='center', va='center', 
                color='white', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, len(colors_rgb))
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Цветовая схема "Огонь" для анализа QLF', fontsize=14, pad=20)
    
    return fig

def create_damage_chart(stats):
    """Создание круговой диаграммы поражения на основе q0-q4"""
    import plotly.graph_objects as go
    
    labels = ['Q4 (Delta > 90)', 'Q3 (75 < Delta ≤ 90)', 
              'Q2 (60 < Delta ≤ 75)', 'Q1 (45 < Delta ≤ 60)', 
              'Здоровые (Delta ≤ 45)']
    
    values = [
        stats.get('q4', 0) * 100,
        stats.get('q3', 0) * 100,
        stats.get('q2', 0) * 100,
        stats.get('q1', 0) * 100,
        max(0, stats.get('q0', 0) * 100 - sum([stats.get(f'q{i}', 0) for i in range(1, 5)]) * 100)
    ]
    
    # Используем цвета из firecolorpalette (только первые 5 для Q1-Q4)
    colors_bgr = [
        (0, 255, 255),  # yellow
        (0, 215, 255),  # gold
        (0, 165, 155),  # orange
        (0, 140, 255),  # darkorange
        (0, 69, 255),   # orangered
    ]
    
    colors_rgb = [f'rgb({c[2]},{c[1]},{c[0]})' for c in colors_bgr]
    colors_rgb.append('rgb(0,0,0)')  # Черный для здоровых областей
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors_rgb,
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title="Распределение поражений по уровням Delta",
        showlegend=True,
        height=400
    )
    
    return fig

def create_roi_control(label, current_value, min_val, max_val, step=10, key_suffix=""):
    """Создание элемента управления ROI с +/- кнопками"""
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        st.markdown(f"**{label}:**")
    
    with col2:
        # Поле ввода
        new_value = st.number_input(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=int(current_value),
            step=step,
            key=f"input_{label}_{key_suffix}",
            label_visibility="collapsed"
        )
    
    with col3:
        # Кнопка -
        if st.button("➖", key=f"minus_{label}_{key_suffix}", 
                    use_container_width=True, help=f"Уменьшить {label} на {step}"):
            new_value = max(min_val, new_value - step)
            st.rerun()
    
    with col4:
        # Кнопка +
        if st.button("➕", key=f"plus_{label}_{key_suffix}", 
                    use_container_width=True, help=f"Увеличить {label} на {step}"):
            new_value = min(max_val, new_value + step)
            st.rerun()
    
    return new_value

def main():
    """Основная функция Streamlit приложения"""
    
    # Заголовок приложения
    st.markdown('<div class="main-header">🦷 QLF Анализатор зубной эмали</div>', 
                unsafe_allow_html=True)
    
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = QLFAnalyzer()
        st.session_state.analysis_done = False
        st.session_state.roi_coords = None
        st.session_state.roi_params = {'x': 0, 'y': 0, 'w': 200, 'h': 200}
        st.session_state.image_info = {'width': 0, 'height': 0}
    
    analyzer = st.session_state.analyzer
    
    # Боковая панель для загрузки и настроек
    with st.sidebar:
        st.header("📁 Загрузка изображения")
        
        uploaded_file = st.file_uploader(
            "Загрузите QLF изображение зубов",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Поддерживаемые форматы: PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            # Загрузка и отображение изображения
            image = analyzer.load_image(uploaded_file)
            img_height, img_width = image.shape[:2]
            
            # Сохраняем информацию об изображении
            st.session_state.image_info = {
                'width': img_width,
                'height': img_height
            }
            
            # Если ROI еще не инициализирована или параметры не соответствуют размеру изображения
            if (st.session_state.roi_params['w'] > img_width or 
                st.session_state.roi_params['h'] > img_height or
                st.session_state.roi_params['x'] == 0 and st.session_state.roi_params['y'] == 0):
                
                # Устанавливаем ROI в центре изображения
                roi_width = min(300, img_width // 2)
                roi_height = min(300, img_height // 2)
                
                st.session_state.roi_params = {
                    'x': max(0, (img_width - roi_width) // 2),
                    'y': max(0, (img_height - roi_height) // 2),
                    'w': roi_width,
                    'h': roi_height
                }
            
            st.header("⚙️ Настройки анализа")
            
            # Выбор ROI с улучшенным интерфейсом
            st.subheader("Выбор области анализа")
            
            # Стилизованная панель управления ROI
            st.markdown('<div class="roi-control-group">', unsafe_allow_html=True)
            
            # X координата
            x_value = create_roi_control(
                "X", 
                st.session_state.roi_params['x'], 
                0, 
                img_width,
                key_suffix="x"
            )
            st.session_state.roi_params['x'] = x_value
            
            # Y координата
            y_value = create_roi_control(
                "Y", 
                st.session_state.roi_params['y'], 
                0, 
                img_height,
                key_suffix="y"
            )
            st.session_state.roi_params['y'] = y_value
            
            st.markdown("---")
            
            # Ширина
            w_value = create_roi_control(
                "Ширина", 
                st.session_state.roi_params['w'], 
                10, 
                img_width,
                key_suffix="w"
            )
            st.session_state.roi_params['w'] = w_value
            
            # Высота
            h_value = create_roi_control(
                "Высота", 
                st.session_state.roi_params['h'], 
                10, 
                img_height,
                key_suffix="h"
            )
            st.session_state.roi_params['h'] = h_value
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Кнопка сброса к центру
            if st.button("🎯 Центрировать ROI", use_container_width=True):
                roi_width = min(300, img_width // 2)
                roi_height = min(300, img_height // 2)
                
                st.session_state.roi_params = {
                    'x': max(0, (img_width - roi_width) // 2),
                    'y': max(0, (img_height - roi_height) // 2),
                    'w': roi_width,
                    'h': roi_height
                }
                st.rerun()
            
            # Информация о ROI
            st.markdown(f"""
            <div class="result-box">
            <b>Параметры ROI:</b><br>
            Позиция: ({st.session_state.roi_params['x']}, {st.session_state.roi_params['y']})<br>
            Размер: {st.session_state.roi_params['w']}×{st.session_state.roi_params['h']}<br>
            Площадь: {st.session_state.roi_params['w'] * st.session_state.roi_params['h']} пикс.
            </div>
            """, unsafe_allow_html=True)
            
            # Предупреждение, если ROI выходит за границы
            if (x_value + w_value > img_width or y_value + h_value > img_height):
                st.warning("⚠️ ROI выходит за границы изображения!")
            
            # Кнопка запуска анализа
            st.markdown("---")
            if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
                # Сохраняем ROI координаты
                roi_coords = (
                    st.session_state.roi_params['x'],
                    st.session_state.roi_params['y'],
                    st.session_state.roi_params['w'],
                    st.session_state.roi_params['h']
                )
                st.session_state.roi_coords = roi_coords
                
                with st.spinner("Выполняется анализ..."):
                    # Анализ ROI
                    st.info("Анализ выбранной области...")
                    roi_stats, roi_viz = analyzer.analyze_roi(roi_coords)
                    
                    # Анализ всех зубов
                    st.info("Анализ всех зубов...")
                    all_stats, all_viz = analyzer.analyze_all_teeth()
                    
                    if roi_stats is not None and all_stats is not None:
                        st.session_state.analysis_done = True
                        st.success("Анализ завершен успешно!")
                        st.rerun()
                    else:
                        st.error("Ошибка при выполнении анализа. Проверьте параметры.")
    
    # Основная область для изображения и результатов
    if uploaded_file is not None:
        if not st.session_state.analysis_done:
            # Показываем изображение с ROI для выбора
            st.markdown('<div class="sub-header">📸 Выбор области анализа</div>', 
                        unsafe_allow_html=True)
            
            # Создаем изображение с ROI
            roi_preview = image.copy()
            x, y, w, h = (
                st.session_state.roi_params['x'],
                st.session_state.roi_params['y'],
                st.session_state.roi_params['w'],
                st.session_state.roi_params['h']
            )
            
            # Рисуем ROI на изображении (зеленая рамка)
            cv2.rectangle(roi_preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Добавляем информацию о координатах на изображение
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(roi_preview, f"ROI", 
                       (x + 5, y + 20), font, 0.6, (0, 255, 0), 2)
            cv2.putText(roi_preview, f"{w}×{h}", 
                       (x + 5, y + h - 10), font, 0.5, (0, 255, 0), 1)
            
            # Показываем изображение с возможностью прокрутки
            st.markdown(f"**Изображение:** {img_width}×{img_height} пикселей")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(analyzer.convert_bgr_to_rgb(roi_preview), 
                    caption="Выделенная область для анализа (ROI)", 
                    use_column_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Информация о размерах
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Размер изображения", f"{img_width}×{img_height}")
            with col2:
                roi_area = w * h
                img_area = img_width * img_height
                st.metric("Покрытие ROI", f"{roi_area/img_area*100:.1f}%")
            
            # Информационный блок
            with st.expander("📋 Информация о методе QLF", expanded=False):
                st.markdown("""
                **Метод количественной светоиндуцированной флуоресценции (QLF)** позволяет:
                - Обнаруживать ранние кариозные поражения
                - Оценивать глубину и площадь деминерализации эмали
                - Контролировать эффективность реминерализующей терапии
                
                **Используемые алгоритмы:**
                - `select_tooth_Analise()`: Анализ выбранной области (ROI)
                - `tooth_Analise()`: Анализ всех зубов с автоматической сегментацией
                - `find_conturs()`: Сегментация зубов по контурам
                
                **Цветовая схема (firecolorpalette):**
                - Желтый: Delta > 120 (максимальное поражение)
                - Золотой: 90 < Delta ≤ 120
                - Оранжевый: 75 < Delta ≤ 90
                - Темно-оранжевый: 60 < Delta ≤ 75
                - Красный: 45 < Delta ≤ 60
                - Синий: Красная флуоресценция (DeltaR+)
                """)
        
        else:
            # Показываем результаты анализа
            st.markdown('<div class="sub-header">📊 Результаты анализа</div>', 
                        unsafe_allow_html=True)
            
            # Вкладки для разных представлений
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Визуализация", 
                "📋 Статистика ROI", 
                "🦷 Статистика всех зубов", 
                "💾 Сохранение"
            ])
            
            with tab1:
                # Визуализация результатов
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Анализ выбранной области (ROI)")
                    if 'roi' in analyzer.results:
                        st.image(analyzer.convert_bgr_to_rgb(analyzer.results['roi']['visualization']), 
                                caption="ROI анализ с цветовой схемой 'Огонь'", 
                                use_column_width=True)
                
                with col2:
                    st.subheader("Анализ всех зубов")
                    if 'all_teeth' in analyzer.results:
                        st.image(analyzer.convert_bgr_to_rgb(analyzer.results['all_teeth']['visualization']), 
                                caption="Анализ всех сегментированных зубов", 
                                use_column_width=True)
                
                # Легенда цветовой схемы
                st.subheader("Цветовая легенда")
                legend_fig = create_fire_legend()
                st.pyplot(legend_fig)
                
                # Диаграммы
                if 'roi' in analyzer.results:
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        roi_chart = create_damage_chart(analyzer.results['roi']['stats'])
                        if roi_chart:
                            st.plotly_chart(roi_chart, use_container_width=True)
            
            with tab2:
                # Детальная статистика ROI
                if 'roi' in analyzer.results:
                    st.subheader("Детальная статистика выбранной области")
                    
                    roi_stats = analyzer.results['roi']['stats']
                    
                    # Показатели в виде метрик
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Q0 (Все поражения)", f"{roi_stats['q0']*100:.1f}%")
                    with col2:
                        st.metric("Q1 (45 < Δ ≤ 60)", f"{roi_stats['q1']*100:.1f}%")
                    with col3:
                        st.metric("Q2 (60 < Δ ≤ 75)", f"{roi_stats['q2']*100:.1f}%")
                    with col4:
                        st.metric("Q3 (75 < Δ ≤ 90)", f"{roi_stats['q3']*100:.1f}%")
                    with col5:
                        st.metric("Q4 (Δ > 90)", f"{roi_stats['q4']*100:.1f}%")
                    
                    # Таблица с дополнительными метриками
                    st.subheader("Дополнительные метрики QLF")
                    
                    metrics_data = [
                        {"Метрика": "mean_deltaR", "Значение": f"{roi_stats['mean_deltaR']:.2f}", "Описание": "Среднее изменение красного канала"},
                        {"Метрика": "mean_deltaF", "Значение": f"{roi_stats['mean_deltaF']:.2f}", "Описание": "Среднее изменение флуоресценции"},
                        {"Метрика": "changing_the_ratio_of_red_and_green_fluorescence", 
                         "Значение": f"{roi_stats['changing_the_ratio_of_red_and_green_fluorescence']:.2f}", 
                         "Описание": "Изменение соотношения красной и зеленой флуоресценции"},
                        {"Метрика": "max_changing_the_ratio_of_red_and_green_fluorescence", 
                         "Значение": f"{roi_stats['max_changing_the_ratio_of_red_and_green_fluorescence']:.2f}", 
                         "Описание": "Максимальное изменение соотношения"},
                        {"Метрика": "average_fluorescence_loss", 
                         "Значение": f"{roi_stats['average_fluorescence_loss']:.2f}", 
                         "Описание": "Средняя потеря флуоресценции"},
                        {"Метрика": "maximum_fluorescence_loss", 
                         "Значение": f"{roi_stats['maximum_fluorescence_loss']:.2f}", 
                         "Описание": "Максимальная потеря флуоресценции"}
                    ]
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Информация о ROI
                    st.subheader("Информация о выбранной области")
                    roi_coords = analyzer.results['roi']['stats']['coords']
                    st.markdown(f"""
                    <div class="result-box">
                    <b>Координаты ROI:</b> X={roi_coords[0]}, Y={roi_coords[1]}<br>
                    <b>Размер ROI:</b> {roi_coords[2]}×{roi_coords[3]} пикселей<br>
                    <b>Площадь ROI:</b> {roi_coords[2] * roi_coords[3]} пикселей
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                # Статистика по всем зубам
                if 'all_teeth' in analyzer.results:
                    st.subheader("Статистика по всем сегментированным зубам")
                    
                    all_stats = analyzer.results['all_teeth']['stats']
                    
                    # Сводные метрики
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Всего контуров", all_stats['total_contours'])
                    with col2:
                        st.metric("Q0 (Все поражения)", f"{all_stats['q0']*100:.1f}%")
                    with col3:
                        st.metric("Q1 (45 < Δ ≤ 60)", f"{all_stats['q1']*100:.1f}%")
                    with col4:
                        st.metric("Q2 (60 < Δ ≤ 75)", f"{all_stats['q2']*100:.1f}%")
                    with col5:
                        st.metric("Q3 (75 < Δ ≤ 90)", f"{all_stats['q3']*100:.1f}%")
                    
                    # Отдельно Q4
                    st.metric("Q4 (Δ > 90)", f"{all_stats['q4']*100:.1f}%")
                    
                    # Таблица с дополнительными метриками для всех зубов
                    st.subheader("Общие метрики QLF для всех зубов")
                    
                    all_metrics_data = [
                        {"Метрика": "mean_deltaR", "Значение": f"{all_stats['mean_deltaR']:.2f}", "Описание": "Среднее изменение красного канала"},
                        {"Метрика": "mean_deltaF", "Значение": f"{all_stats['mean_deltaF']:.2f}", "Описание": "Среднее изменение флуоресценции"},
                        {"Метрика": "changing_the_ratio_of_red_and_green_fluorescence", 
                         "Значение": f"{all_stats['changing_the_ratio_of_red_and_green_fluorescence']:.2f}", 
                         "Описание": "Изменение соотношения красной и зеленой флуоресценции"},
                        {"Метрика": "max_changing_the_ratio_of_red_and_green_fluorescence", 
                         "Значение": f"{all_stats['max_changing_the_ratio_of_red_and_green_fluorescence']:.2f}", 
                         "Описание": "Максимальное изменение соотношения"},
                        {"Метрика": "average_fluorescence_loss", 
                         "Значение": f"{all_stats['average_fluorescence_loss']:.2f}", 
                         "Описание": "Средняя потеря флуоресценции"},
                        {"Метрика": "maximum_fluorescence_loss", 
                         "Значение": f"{all_stats['maximum_fluorescence_loss']:.2f}", 
                         "Описание": "Максимальная потеря флуоресценции"}
                    ]
                    
                    all_metrics_df = pd.DataFrame(all_metrics_data)
                    st.dataframe(all_metrics_df, use_container_width=True, hide_index=True)
            
            with tab4:
                # Сохранение результатов
                st.subheader("Сохранение результатов анализа")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Создание временных файлов
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Сохранение изображений
                    if 'roi' in analyzer.results:
                        roi_img_path = os.path.join(tmpdir, f"roi_analysis_{timestamp}.png")
                        cv2.imwrite(roi_img_path, analyzer.results['roi']['visualization'])
                    
                    if 'all_teeth' in analyzer.results:
                        all_teeth_img_path = os.path.join(tmpdir, f"all_teeth_analysis_{timestamp}.png")
                        cv2.imwrite(all_teeth_img_path, analyzer.results['all_teeth']['visualization'])
                    
                    # Создание текстового отчета
                    report_content = f"""QLF АНАЛИЗ ЗУБНОЙ ЭМАЛИ (Используются алгоритмы dantistUtils)
Отчет сгенерирован: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Файл изображения: {uploaded_file.name}
Используемая цветовая палитра: firecolorpalette

"""
                    
                    if 'roi' in analyzer.results:
                        roi_stats = analyzer.results['roi']['stats']
                        roi_coords = analyzer.results['roi']['stats'].get('coords', (0, 0, 0, 0))
                        report_content += f"""
1. АНАЛИЗ ВЫБРАННОЙ ОБЛАСТИ (ROI):
   Координаты: X={roi_coords[0]}, Y={roi_coords[1]}, Ширина={roi_coords[2]}, Высота={roi_coords[3]}
   
   Распределение поражений по уровням Delta:
   - Q0 (Все поражения Delta > 45): {roi_stats['q0']*100:.1f}%
   - Q1 (45 < Delta ≤ 60): {roi_stats['q1']*100:.1f}%
   - Q2 (60 < Delta ≤ 75): {roi_stats['q2']*100:.1f}%
   - Q3 (75 < Delta ≤ 90): {roi_stats['q3']*100:.1f}%
   - Q4 (Delta > 90): {roi_stats['q4']*100:.1f}%
   
   Метрики QLF:
   - mean_deltaR: {roi_stats['mean_deltaR']:.2f}
   - mean_deltaF: {roi_stats['mean_deltaF']:.2f}
   - changing_the_ratio_of_red_and_green_fluorescence: {roi_stats['changing_the_ratio_of_red_and_green_fluorescence']:.2f}
   - max_changing_the_ratio_of_red_and_green_fluorescence: {roi_stats['max_changing_the_ratio_of_red_and_green_fluorescence']:.2f}
   - average_fluorescence_loss: {roi_stats['average_fluorescence_loss']:.2f}
   - maximum_fluorescence_loss: {roi_stats['maximum_fluorescence_loss']:.2f}

"""
                    
                    if 'all_teeth' in analyzer.results:
                        all_stats = analyzer.results['all_teeth']['stats']
                        report_content += f"""
2. АНАЛИЗ ВСЕХ ЗУБОВ (АВТОМАТИЧЕСКАЯ СЕГМЕНТАЦИЯ):
   Всего обнаружено контуров зубов: {all_stats['total_contours']}
   
   Распределение поражений по уровням Delta:
   - Q0 (Все поражения Delta > 45): {all_stats['q0']*100:.1f}%
   - Q1 (45 < Delta ≤ 60): {all_stats['q1']*100:.1f}%
   - Q2 (60 < Delta ≤ 75): {all_stats['q2']*100:.1f}%
   - Q3 (75 < Delta ≤ 90): {all_stats['q3']*100:.1f}%
   - Q4 (Delta > 90): {all_stats['q4']*100:.1f}%
   
   Метрики QLF:
   - mean_deltaR: {all_stats['mean_deltaR']:.2f}
   - mean_deltaF: {all_stats['mean_deltaF']:.2f}
   - changing_the_ratio_of_red_and_green_fluorescence: {all_stats['changing_the_ratio_of_red_and_green_fluorescence']:.2f}
   - max_changing_the_ratio_of_red_and_green_fluorescence: {all_stats['max_changing_the_ratio_of_red_and_green_fluorescence']:.2f}
   - average_fluorescence_loss: {all_stats['average_fluorescence_loss']:.2f}
   - maximum_fluorescence_loss: {all_stats['maximum_fluorescence_loss']:.2f}
"""
                    
                    report_path = os.path.join(tmpdir, f"qlf_report_{timestamp}.txt")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    # JSON данные
                    json_data = {
                        'timestamp': timestamp,
                        'original_filename': uploaded_file.name,
                        'analysis_method': 'dantistUtils algorithms',
                        'color_palette': 'firecolorpalette'
                    }
                    
                    if 'roi' in analyzer.results:
                        json_data['roi_analysis'] = analyzer.results['roi']['stats']
                    
                    if 'all_teeth' in analyzer.results:
                        json_data['all_teeth_analysis'] = analyzer.results['all_teeth']['stats']
                    
                    json_path = os.path.join(tmpdir, f"qlf_data_{timestamp}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)
                    
                    # Кнопки для скачивания
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'roi' in analyzer.results:
                            with open(roi_img_path, "rb") as f:
                                st.download_button(
                                    label="📥 ROI изображение",
                                    data=f,
                                    file_name=f"roi_analysis_{timestamp}.png",
                                    mime="image/png"
                                )
                    
                    with col2:
                        if 'all_teeth' in analyzer.results:
                            with open(all_teeth_img_path, "rb") as f:
                                st.download_button(
                                    label="📥 Все зубы",
                                    data=f,
                                    file_name=f"all_teeth_analysis_{timestamp}.png",
                                    mime="image/png"
                                )
                    
                    with col3:
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Текстовый отчет",
                                data=f,
                                file_name=f"qlf_report_{timestamp}.txt",
                                mime="text/plain"
                            )
                    
                    with col4:
                        with open(json_path, "rb") as f:
                            st.download_button(
                                label="📥 JSON данные",
                                data=f,
                                file_name=f"qlf_data_{timestamp}.json",
                                mime="application/json"
                            )
                    
                    # Создание ZIP архива
                    import zipfile
                    zip_path = os.path.join(tmpdir, f"qlf_analysis_{timestamp}.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        files_to_zip = []
                        if 'roi' in analyzer.results:
                            files_to_zip.append(roi_img_path)
                        if 'all_teeth' in analyzer.results:
                            files_to_zip.append(all_teeth_img_path)
                        files_to_zip.extend([report_path, json_path])
                        
                        for file in files_to_zip:
                            zipf.write(file, os.path.basename(file))
                    
                    st.subheader("Скачать все результаты")
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="📦 Скачать все файлы (ZIP)",
                            data=f,
                            file_name=f"qlf_analysis_{timestamp}.zip",
                            mime="application/zip"
                        )
    
    elif uploaded_file is None:
        # Экран приветствия
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style="text-align: center;">
            <h2>🦷 QLF Анализатор</h2>
            <p>Минималистичный интерфейс с удобным выбором ROI</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>👋 Добро пожаловать в QLF Анализатор!</h3>
        <p><b>Простой и удобный интерфейс:</b></p>
        <ol>
            <li><b>Автоматическое центрирование</b> - ROI устанавливается в центре изображения</li>
            <li><b>Минималистичное управление</b> - только поля ввода и кнопки +/-</li>
            <li><b>Шаг изменения 10 пикселей</b> - для точной настройки</li>
            <li><b>Полноразмерный просмотр</b> - видно всю область ROI</li>
            <li><b>Одна кнопка "Центрировать"</b> - быстро вернуть ROI в центр</li>
        </ol>
        <p><b>Чтобы начать:</b></p>
        <ol>
            <li>Загрузите QLF изображение зубов</li>
            <li>Настройте ROI с помощью кнопок +/- или ввода значений</li>
            <li>Нажмите "Запустить анализ"</li>
            <li>Просмотрите результаты в четырех вкладках</li>
        </ol>
        <p><b>Поддерживаемые форматы:</b> PNG, JPG, JPEG, TIFF, BMP</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Проверка наличия необходимых библиотек
    try:
        import cv2
        import streamlit
        import plotly
    except ImportError as e:
        st.error(f"Отсутствует необходимая библиотека: {e}")
        st.info("Установите необходимые библиотеки:")
        st.code("pip install streamlit opencv-python numpy matplotlib pillow pandas plotly")
    else:
        main()