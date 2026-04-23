import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pandas as pd
from datetime import datetime
import json
import tempfile
import os
from skimage import measure, morphology
import plotly.graph_objects as go
import plotly.express as px

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
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B6CA8;
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
    
    def preprocess_image(self, img):
        """Предобработка изображения"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        filtered = cv2.medianBlur(enhanced, 5)
        return filtered
    
    def segment_all_teeth(self, progress_bar=None):
        """Сегментация всех зубов на изображении"""
        if self.image is None:
            return []
        
        processed = self.preprocess_image(self.image)
        binary = cv2.adaptiveThreshold(processed, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3,3), np.uint8)
        cleaned = morphology.opening(binary, kernel)
        cleaned = morphology.closing(cleaned, kernel)
        
        cleaned_bool = cleaned.astype(bool)
        cleaned_bool = morphology.remove_small_objects(cleaned_bool, min_size=500)
        cleaned = cleaned_bool.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        self.segmented_teeth = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:
                    tooth_mask = np.zeros_like(cleaned)
                    cv2.drawContours(tooth_mask, [contour], -1, 255, -1)
                    
                    tooth_region = cv2.bitwise_and(self.image, self.image, 
                                                  mask=tooth_mask)
                    
                    self.segmented_teeth.append({
                        'id': len(self.segmented_teeth) + 1,
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'mask': tooth_mask,
                        'region': tooth_region[y:y+h, x:x+w],
                        'full_region': tooth_region,
                        'area_pixels': int(area),
                        'center': (x + w//2, y + h//2)
                    })
                    
                    if progress_bar:
                        progress_bar.progress((i + 1) / len(contours))
        
        return self.segmented_teeth
    
    def analyze_enamel_fluorescence(self, region):
        """Анализ флуоресценции эмали"""
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region
        
        normalized = cv2.normalize(gray_region, None, 0, 255, cv2.NORM_MINMAX)
        health_map = normalized.astype(float) / 255.0
        damage_map = 1.0 - health_map
        
        damage_levels = np.zeros_like(damage_map, dtype=np.uint8)
        damage_levels[damage_map <= 0.20] = 0
        damage_levels[(damage_map > 0.20) & (damage_map <= 0.50)] = 1
        damage_levels[(damage_map > 0.50) & (damage_map <= 0.75)] = 2
        damage_levels[damage_map > 0.75] = 3
        
        total_pixels = damage_map.size
        level_areas = [
            int(np.sum(damage_levels == 0)),
            int(np.sum(damage_levels == 1)),
            int(np.sum(damage_levels == 2)),
            int(np.sum(damage_levels == 3))
        ]
        
        level_percentages = [float(area / total_pixels * 100) for area in level_areas]
        
        level_weights = [0.1, 0.35, 0.63, 0.88]
        weighted_damage = sum(p * w for p, w in zip(level_percentages, level_weights)) / 100
        
        stats = {
            'total_pixels': int(total_pixels),
            'level_areas': level_areas,
            'level_percentages': level_percentages,
            'level_0_percent': float(level_percentages[0]),
            'level_1_percent': float(level_percentages[1]),
            'level_2_percent': float(level_percentages[2]),
            'level_3_percent': float(level_percentages[3]),
            'mean_damage': float(np.mean(damage_map) * 100),
            'median_damage': float(np.median(damage_map) * 100),
            'max_damage': float(np.max(damage_map) * 100),
            'std_damage': float(np.std(damage_map) * 100),
            'weighted_damage_score': float(weighted_damage * 100),
        }
        
        return stats, damage_levels
    
    def apply_fire_colormap(self, damage_levels):
        """Применение цветовой схемы 'огонь и угли'"""
        colors = np.array([
            [0, 0, 0],           # 0% - черный
            [139, 69, 19],       # 20% - темно-коричневый
            [205, 133, 63],      # 50% - перу
            [255, 165, 0],       # 75% - оранжевый
            [255, 255, 0],       # 100% - желтый
        ]) / 255.0
        
        colored = np.zeros((*damage_levels.shape, 3))
        for level in range(4):
            mask = damage_levels == level
            colored[mask] = colors[level]
        
        return (colored * 255).astype(np.uint8)
    
    def analyze_roi(self, roi_coords):
        """Анализ выбранной пользователем области"""
        x, y, w, h = roi_coords
        roi_region = self.image[y:y+h, x:x+w]
        
        roi_stats, damage_levels = self.analyze_enamel_fluorescence(roi_region)
        roi_colored = self.apply_fire_colormap(damage_levels)
        
        roi_visualization = self.image.copy()
        cv2.rectangle(roi_visualization, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        roi_visualization[y:y+h, x:x+w] = roi_colored
        
        self.results['roi'] = {
            'stats': roi_stats,
            'visualization': roi_visualization,
            'roi_colored': roi_colored,
            'coords': roi_coords,
            'region': roi_region
        }
        
        return roi_stats, roi_visualization
    
    def analyze_all_teeth(self):
        """Анализ всех сегментированных зубов"""
        all_teeth_stats = []
        all_teeth_visualization = self.image.copy()
        
        for tooth in self.segmented_teeth:
            x, y, w, h = tooth['bbox']
            tooth_region = tooth['region']
            
            tooth_stats, damage_levels = self.analyze_enamel_fluorescence(tooth_region)
            tooth_colored = self.apply_fire_colormap(damage_levels)
            
            tooth_mask_resized = cv2.resize(tooth['mask'][y:y+h, x:x+w].astype(np.uint8), 
                                          (w, h))
            
            mask_3channel = np.stack([tooth_mask_resized]*3, axis=-1) > 0
            all_teeth_visualization[y:y+h, x:x+w][mask_3channel] = \
                tooth_colored[mask_3channel]
            
            cv2.rectangle(all_teeth_visualization, (x, y), (x+w, y+h), 
                         (255, 0, 0), 2)
            
            avg_damage = tooth_stats['mean_damage']
            cv2.putText(all_teeth_visualization, f"{avg_damage:.1f}%", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 2)
            
            all_teeth_stats.append({
                'tooth_id': tooth['id'],
                'bbox': tooth['bbox'],
                'area_pixels': tooth['area_pixels'],
                **tooth_stats
            })
        
        total_stats = {
            'total_teeth': len(all_teeth_stats),
            'avg_mean_damage': float(np.mean([s['mean_damage'] for s in all_teeth_stats])),
            'max_damage': float(np.max([s['max_damage'] for s in all_teeth_stats])),
            'avg_weighted_score': float(np.mean([s['weighted_damage_score'] for s in all_teeth_stats])),
            'teeth_stats': all_teeth_stats
        }
        
        self.results['all_teeth'] = {
            'stats': total_stats,
            'visualization': all_teeth_visualization,
            'individual_stats': all_teeth_stats
        }
        
        return total_stats, all_teeth_visualization

def create_fire_legend():
    """Создание легенды цветовой схемы"""
    fig, ax = plt.subplots(figsize=(10, 1))
    colors = ['black', 'saddlebrown', 'peru', 'orange', 'yellow']
    labels = ['0-20%: Здоровые', '21-50%: Начальное', 
              '51-75%: Среднее', '>75%: Сильное', 'Пламя →']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.barh(0, 1, left=i, color=color, edgecolor='black', height=0.8)
        ax.text(i + 0.5, 0, label, ha='center', va='center', 
                color='white' if color != 'yellow' else 'black', fontsize=10)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Цветовая схема "Огонь и угли"', fontsize=12, pad=20)
    
    return fig

def create_damage_chart(roi_stats, title):
    """Создание круговой диаграммы поражения"""
    labels = ['Здоровые (0-20%)', 'Начальные (21-50%)', 
              'Средние (51-75%)', 'Сильные (>75%)']
    
    values = [
        roi_stats['level_0_percent'],
        roi_stats['level_1_percent'],
        roi_stats['level_2_percent'],
        roi_stats['level_3_percent']
    ]
    
    colors = ['black', 'saddlebrown', 'peru', 'orange']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors,
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400
    )
    
    return fig

def main():
    """Основная функция Streamlit приложения"""
    
    # Заголовок приложения
    st.markdown('<div class="main-header">🦷 QLF Анализатор зубной эмали</div>', 
                unsafe_allow_html=True)
    
    # Информационный блок
    with st.expander("ℹ️ Информация о методе QLF", expanded=False):
        st.markdown("""
        **Метод количественной светоиндуцированной флуоресценции (QLF)** позволяет:
        - Обнаруживать ранние кариозные поражения
        - Оценивать глубину и площадь деминерализации эмали
        - Контролировать эффективность реминерализующей терапии
        
        **Принцип работы**: Здоровая эмаль флуоресцирует под синим светом (405 нм).
        Деминерализованные участки теряют флуоресценцию, что визуализируется как 
        тёмные области на изображении.
        """)
    
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = QLFAnalyzer()
        st.session_state.analysis_done = False
        st.session_state.roi_coords = None
    
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
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Загруженное изображение", 
                    use_column_width=True)
            
            st.header("⚙️ Настройки анализа")
            
            # Настройка ROI
            st.subheader("Выбор области анализа")
            img_height, img_width = image.shape[:2]
            
            col1, col2 = st.columns(2)
            with col1:
                roi_x = st.slider("X координата", 0, img_width, img_width//4, key="roi_x")
                roi_w = st.slider("Ширина", 10, img_width//2, img_width//3, key="roi_w")
            with col2:
                roi_y = st.slider("Y координата", 0, img_height, img_height//4, key="roi_y")
                roi_h = st.slider("Высота", 10, img_height//2, img_height//3, key="roi_h")
            
            roi_coords = (roi_x, roi_y, roi_w, roi_h)
            st.session_state.roi_coords = roi_coords
            
            # Показать предварительный ROI
            roi_preview = image.copy()
            cv2.rectangle(roi_preview, (roi_x, roi_y), 
                         (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
            st.image(cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB), 
                    caption="Предпросмотр выбранной области", 
                    use_column_width=True)
            
            # Кнопка запуска анализа
            if st.button("🚀 Запустить анализ", type="primary"):
                with st.spinner("Выполняется анализ..."):
                    # Сегментация зубов
                    progress_bar = st.progress(0)
                    st.info("Сегментация зубов...")
                    teeth = analyzer.segment_all_teeth(progress_bar)
                    
                    # Анализ ROI
                    st.info("Анализ выбранной области...")
                    roi_stats, roi_viz = analyzer.analyze_roi(roi_coords)
                    
                    # Анализ всех зубов
                    st.info("Анализ всех зубов...")
                    all_stats, all_viz = analyzer.analyze_all_teeth()
                    
                    st.session_state.analysis_done = True
                    st.success("Анализ завершен!")
    
    # Основная область для результатов
    if uploaded_file is not None and st.session_state.analysis_done:
        st.markdown('<div class="sub-header">📊 Результаты анализа</div>', 
                    unsafe_allow_html=True)
        
        # Вкладки для разных представлений
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Визуализация", 
            "📋 Статистика ROI", 
            "🦷 Статистика зубов", 
            "💾 Сохранение"
        ])
        
        with tab1:
            # Визуализация результатов
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Анализ выбранной области (ROI)")
                st.image(cv2.cvtColor(analyzer.results['roi']['visualization'], cv2.COLOR_BGR2RGB), 
                        caption="ROI анализ с цветовой схемой", 
                        use_column_width=True)
            
            with col2:
                st.subheader("Анализ всех зубов")
                st.image(cv2.cvtColor(analyzer.results['all_teeth']['visualization'], cv2.COLOR_BGR2RGB), 
                        caption="Анализ всех сегментированных зубов", 
                        use_column_width=True)
            
            # Легенда цветовой схемы
            st.subheader("Цветовая легенда")
            legend_fig = create_fire_legend()
            st.pyplot(legend_fig)
            
            # Диаграммы
            col3, col4 = st.columns(2)
            
            with col3:
                if 'roi' in analyzer.results:
                    roi_chart = create_damage_chart(
                        analyzer.results['roi']['stats'], 
                        "Распределение поражений в ROI"
                    )
                    st.plotly_chart(roi_chart, use_container_width=True)
            
            with col4:
                # Упрощенная диаграмма для всех зубов
                if 'all_teeth' in analyzer.results and analyzer.results['all_teeth']['individual_stats']:
                    all_stats = analyzer.results['all_teeth']['individual_stats']
                    avg_levels = [
                        np.mean([tooth[f'level_{i}_percent'] for tooth in all_stats])
                        for i in range(4)
                    ]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Здоровые', 'Начальные', 'Средние', 'Сильные'],
                        values=avg_levels,
                        hole=.3,
                        marker_colors=['black', 'saddlebrown', 'peru', 'orange']
                    )])
                    fig.update_layout(title="Среднее распределение по всем зубам")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Детальная статистика ROI
            st.subheader("Детальная статистика выбранной области")
            
            roi_stats = analyzer.results['roi']['stats']
            
            # Показатели в виде метрик
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Среднее поражение", f"{roi_stats['mean_damage']:.1f}%")
            with col2:
                st.metric("Максимальное поражение", f"{roi_stats['max_damage']:.1f}%")
            with col3:
                st.metric("Взвешенный индекс", f"{roi_stats['weighted_damage_score']:.1f}")
            with col4:
                st.metric("Всего пикселей", f"{roi_stats['total_pixels']:,}")
            
            # Таблица распределения по уровням
            st.subheader("Распределение по уровням поражения")
            roi_levels_df = pd.DataFrame({
                'Уровень поражения': ['Здоровые (0-20%)', 'Начальные (21-50%)', 
                                     'Средние (51-75%)', 'Сильные (>75%)'],
                'Площадь (пиксели)': roi_stats['level_areas'],
                'Процент площади': [f"{p:.1f}%" for p in roi_stats['level_percentages']]
            })
            st.dataframe(roi_levels_df, use_container_width=True, hide_index=True)
            
            # Дополнительная статистика
            st.subheader("Дополнительные показатели")
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(f"""
                <div class="result-box">
                <b>Медианное поражение:</b> {roi_stats['median_damage']:.1f}%<br>
                <b>Стандартное отклонение:</b> {roi_stats['std_damage']:.1f}%<br>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                roi_coords = analyzer.results['roi']['coords']
                st.markdown(f"""
                <div class="result-box">
                <b>Координаты ROI:</b> X={roi_coords[0]}, Y={roi_coords[1]}<br>
                <b>Размер ROI:</b> {roi_coords[2]}×{roi_coords[3]} пикселей<br>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Статистика по всем зубам
            st.subheader("Статистика по всем сегментированным зубам")
            
            all_stats = analyzer.results['all_teeth']['stats']
            individual_stats = analyzer.results['all_teeth']['individual_stats']
            
            # Сводные метрики
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего зубов", all_stats['total_teeth'])
            with col2:
                st.metric("Среднее поражение", f"{all_stats['avg_mean_damage']:.1f}%")
            with col3:
                st.metric("Макс. поражение", f"{all_stats['max_damage']:.1f}%")
            
            # Детальная таблица по зубам
            st.subheader("Детальная статистика по каждому зубу")
            
            teeth_data = []
            for tooth in individual_stats:
                teeth_data.append({
                    'ID зуба': tooth['tooth_id'],
                    'Координаты': f"({tooth['bbox'][0]}, {tooth['bbox'][1]})",
                    'Размер': f"{tooth['bbox'][2]}×{tooth['bbox'][3]}",
                    'Площадь': tooth['area_pixels'],
                    'Среднее поражение': f"{tooth['mean_damage']:.1f}%",
                    'Макс. поражение': f"{tooth['max_damage']:.1f}%",
                    'Здоровые': f"{tooth['level_0_percent']:.1f}%",
                    'Начальные': f"{tooth['level_1_percent']:.1f}%",
                    'Средние': f"{tooth['level_2_percent']:.1f}%",
                    'Сильные': f"{tooth['level_3_percent']:.1f}%"
                })
            
            teeth_df = pd.DataFrame(teeth_data)
            st.dataframe(teeth_df, use_container_width=True)
            
            # Визуализация распределения поражений
            if len(teeth_data) > 0:
                st.subheader("Распределение поражений по зубам")
                
                # Преобразование данных для графика
                plot_data = []
                for tooth in teeth_data:
                    for level in ['Здоровые', 'Начальные', 'Средние', 'Сильные']:
                        plot_data.append({
                            'Зуб': f"Зуб {tooth['ID зуба']}",
                            'Уровень поражения': level,
                            'Процент': float(tooth[level].replace('%', ''))
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                fig = px.bar(
                    plot_df,
                    x='Зуб',
                    y='Процент',
                    color='Уровень поражения',
                    title="Распределение уровней поражения по зубам",
                    color_discrete_map={
                        'Здоровые': 'black',
                        'Начальные': 'saddlebrown',
                        'Средние': 'peru',
                        'Сильные': 'orange'
                    }
                )
                fig.update_layout(barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Сохранение результатов
            st.subheader("Сохранение результатов анализа")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Создание временных файлов
            with tempfile.TemporaryDirectory() as tmpdir:
                # Сохранение изображений
                roi_img_path = os.path.join(tmpdir, f"roi_analysis_{timestamp}.png")
                all_teeth_img_path = os.path.join(tmpdir, f"all_teeth_analysis_{timestamp}.png")
                
                cv2.imwrite(roi_img_path, analyzer.results['roi']['visualization'])
                cv2.imwrite(all_teeth_img_path, analyzer.results['all_teeth']['visualization'])
                
                # Создание текстового отчета
                report_content = f"""QLF АНАЛИЗ ЗУБНОЙ ЭМАЛИ
Отчет сгенерирован: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Файл изображения: {uploaded_file.name}

1. АНАЛИЗ ВЫДЕЛЕННОЙ ОБЛАСТИ (ROI):
   Координаты: X={analyzer.results['roi']['coords'][0]}, Y={analyzer.results['roi']['coords'][1]}, Ширина={analyzer.results['roi']['coords'][2]}, Высота={analyzer.results['roi']['coords'][3]}
   Всего пикселей: {analyzer.results['roi']['stats']['total_pixels']}
   
   Распределение поражений:
   - Здоровые области (0-20%): {analyzer.results['roi']['stats']['level_0_percent']:.1f}%
   - Начальное поражение (21-50%): {analyzer.results['roi']['stats']['level_1_percent']:.1f}%
   - Среднее поражение (51-75%): {analyzer.results['roi']['stats']['level_2_percent']:.1f}%
   - Сильное поражение (>75%): {analyzer.results['roi']['stats']['level_3_percent']:.1f}%
   
   Статистика:
   - Среднее поражение: {analyzer.results['roi']['stats']['mean_damage']:.1f}%
   - Медианное поражение: {analyzer.results['roi']['stats']['median_damage']:.1f}%
   - Максимальное поражение: {analyzer.results['roi']['stats']['max_damage']:.1f}%
   - Взвешенный индекс поражения: {analyzer.results['roi']['stats']['weighted_damage_score']:.1f}

2. АНАЛИЗ ВСЕХ ЗУБОВ:
   Всего обнаружено зубов: {analyzer.results['all_teeth']['stats']['total_teeth']}
   
   Сводная статистика:
   - Среднее поражение по всем зубам: {analyzer.results['all_teeth']['stats']['avg_mean_damage']:.1f}%
   - Максимальное поражение: {analyzer.results['all_teeth']['stats']['max_damage']:.1f}%
   - Средний взвешенный индекс: {analyzer.results['all_teeth']['stats']['avg_weighted_score']:.1f}
   
   Детальная статистика по зубам:
"""
                
                for tooth in analyzer.results['all_teeth']['individual_stats']:
                    report_content += f"""
   Зуб {tooth['tooth_id']}:
      Координаты: ({tooth['bbox'][0]}, {tooth['bbox'][1]})
      Размер: {tooth['bbox'][2]}×{tooth['bbox'][3]} пикселей
      Площадь: {tooth['area_pixels']} пикселей
      Среднее поражение: {tooth['mean_damage']:.1f}%
      Макс. поражение: {tooth['max_damage']:.1f}%
      Распределение: 
        Здоровые: {tooth['level_0_percent']:.1f}%
        Начальные: {tooth['level_1_percent']:.1f}%
        Средние: {tooth['level_2_percent']:.1f}%
        Сильные: {tooth['level_3_percent']:.1f}%
"""
                
                report_path = os.path.join(tmpdir, f"qlf_report_{timestamp}.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                # JSON данные (сериализуемые)
                json_data = {
                    'timestamp': timestamp,
                    'original_filename': uploaded_file.name,
                    'image_dimensions': f"{img_width}x{img_height}",
                    'roi_analysis': analyzer.results['roi']['stats'],
                    'all_teeth_analysis': analyzer.results['all_teeth']['stats'],
                    'individual_teeth': analyzer.results['all_teeth']['individual_stats']
                }
                
                json_path = os.path.join(tmpdir, f"qlf_data_{timestamp}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                
                # Кнопки для скачивания
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    with open(roi_img_path, "rb") as f:
                        st.download_button(
                            label="📥 ROI изображение",
                            data=f,
                            file_name=f"roi_analysis_{timestamp}.png",
                            mime="image/png"
                        )
                
                with col2:
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
                    for file in [roi_img_path, all_teeth_img_path, report_path, json_path]:
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
            <p>Автоматический анализ QLF-снимков зубной эмали</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>👋 Добро пожаловать в QLF Анализатор!</h3>
        <p>Этот инструмент позволяет анализировать QLF-снимки зубов для оценки состояния эмали.</p>
        <p><b>Чтобы начать:</b></p>
        <ol>
            <li>Загрузите QLF изображение зубов в боковой панели</li>
            <li>Выберите область для анализа (ROI)</li>
            <li>Нажмите кнопку "Запустить анализ"</li>
            <li>Просмотрите результаты и скачайте отчеты</li>
        </ol>
        <p><b>Поддерживаемые форматы:</b> PNG, JPG, JPEG, TIFF, BMP</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Пример интерфейса
        with st.expander("📸 Пример работы анализатора", expanded=False):
            st.markdown("""
            **Пример интерфейса анализа:**
            1. Загрузите изображение зубов
            2. Выберите область для детального анализа
            3. Получите автоматическую сегментацию всех зубов
            4. Просмотрите результаты с цветовой кодировкой
            
            **Цветовая схема "Огонь и угли":**
            - Черный (0-20%): здоровые области
            - Коричневый (21-50%): начальное поражение
            - Оранжевый (51-75%): среднее поражение
            - Желтый (>75%): сильное поражение
            """)

if __name__ == "__main__":
    main()