import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import time
import warnings

from util.decribe import get_texts_app

warnings.filterwarnings('ignore')

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class TextClusterApp:
    def __init__(self):
        self.texts = []
        self.vectorizer = None
        self.X = None
        self.feature_names = []

    def load_sample_data(self):
        """Загрузка примеров текстов для демонстрации"""
        sample_texts = get_texts_app()
        return sample_texts

    def setup_vectorizer(self, method='tfidf', **kwargs):
        """Настройка векторизатора"""
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif method == 'count':
            self.vectorizer = CountVectorizer(**kwargs)
        return self.vectorizer

    def vectorize_texts(self, texts, method='tfidf', max_features=2000, ngram_range=(1, 2)):
        """Векторизация текстов"""
        vectorizer_params = {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'stop_words': None,
            'min_df': 2,
            'max_df': 0.8
        }

        self.setup_vectorizer(method, **vectorizer_params)
        self.X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.X

    def perform_clustering(self, algorithm, X, **params):
        """Выполнение кластеризации"""
        start_time = time.time()

        if algorithm == 'KMeans':
            model = KMeans(
                n_clusters=params.get('n_clusters', 5),
                random_state=params.get('random_state', 42)
            )
            labels = model.fit_predict(X)

        elif algorithm == 'MiniBatchKMeans':
            model = MiniBatchKMeans(
                n_clusters=params.get('n_clusters', 5),
                random_state=params.get('random_state', 42),
                batch_size=512
            )
            labels = model.fit_predict(X)

        elif algorithm == 'DBSCAN':
            model = DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
            labels = model.fit_predict(X)

        elif algorithm == 'HDBSCAN' and HDBSCAN_AVAILABLE:
            model = hdbscan.HDBSCAN(
                min_cluster_size=params.get('min_cluster_size', 10),
                min_samples=params.get('min_samples', 5)
            )
            labels = model.fit_predict(X.toarray() if hasattr(X, 'toarray') else X)

        elif algorithm == 'GaussianMixture':
            model = GaussianMixture(
                n_components=params.get('n_clusters', 5),
                random_state=params.get('random_state', 42)
            )
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            labels = model.fit_predict(X_dense)

        elif algorithm == 'SpectralClustering':
            spectral_params = {
                'n_clusters': params.get('n_clusters', 5),
                'random_state': params.get('random_state', 42),
                'affinity': params.get('affinity', 'rbf')
            }
            if params.get('affinity') == 'nearest_neighbors':
                spectral_params['n_neighbors'] = params.get('n_neighbors', 10)

            model = SpectralClustering(**spectral_params)

            # Для SpectralClustering используем плотные матрицы для некоторых affinity
            if params.get('affinity') == 'cosine' and hasattr(X, 'toarray'):
                X_used = X
            else:
                X_used = X.toarray() if hasattr(X, 'toarray') else X
            labels = model.fit_predict(X_used)

        elif algorithm == 'AgglomerativeClustering':
            model = AgglomerativeClustering(
                n_clusters=params.get('n_clusters', 5),
                linkage=params.get('linkage', 'ward')
            )
            # Для ward linkage нужна плотная матрица
            if params.get('linkage') == 'ward':
                X_used = X.toarray() if hasattr(X, 'toarray') else X
            else:
                X_used = X
            labels = model.fit_predict(X_used)

        else:
            raise ValueError(f"Алгоритм {algorithm} не поддерживается")

        execution_time = time.time() - start_time
        return labels, model, execution_time

    def calculate_metrics(self, X, labels):
        """Вычисление метрик кластеризации"""
        metrics = {}

        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Игнорируем шум (-1) при подсчете кластеров
        valid_labels = labels[labels != -1]
        n_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0

        if n_clusters > 1 and len(valid_labels) > 1:
            try:
                metrics['silhouette'] = silhouette_score(X_dense[labels != -1], valid_labels)
            except:
                metrics['silhouette'] = -1

            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X_dense[labels != -1], valid_labels)
            except:
                metrics['calinski_harabasz'] = -1

            try:
                metrics['davies_bouldin'] = davies_bouldin_score(X_dense[labels != -1], valid_labels)
            except:
                metrics['davies_bouldin'] = float('inf')
        else:
            metrics['silhouette'] = -1
            metrics['calinski_harabasz'] = -1
            metrics['davies_bouldin'] = float('inf')

        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = np.sum(labels == -1)
        metrics['n_points'] = len(labels)

        return metrics

    def get_cluster_keywords(self, labels, n_words=10):
        """Получение ключевых слов для каждого кластера"""
        if self.X is None or len(self.feature_names) == 0:
            return {}

        unique_labels = np.unique(labels)
        cluster_keywords = {}

        for cluster_id in unique_labels:
            if cluster_id == -1:  # Шум
                continue

            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                cluster_keywords[cluster_id] = []
                continue

            # Средние TF-IDF веса для кластера
            if hasattr(self.X, 'toarray'):
                cluster_tfidf = self.X[cluster_indices].mean(axis=0).A1
            else:
                cluster_tfidf = self.X[cluster_indices].mean(axis=0)

            # Топ-N слов
            top_indices = np.argsort(cluster_tfidf)[::-1][:n_words]
            top_words = [(self.feature_names[i], cluster_tfidf[i])
                         for i in top_indices if cluster_tfidf[i] > 0]

            cluster_keywords[cluster_id] = top_words

        return cluster_keywords

    def create_wordclouds(self, labels, n_words=20):
        """Создание облаков слов для кластеров"""
        cluster_keywords = self.get_cluster_keywords(labels, n_words)
        figs = []

        for cluster_id, words_weights in cluster_keywords.items():
            if not words_weights:
                continue

            word_freq = {word: weight for word, weight in words_weights}
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Кластер {cluster_id}', fontsize=16, fontweight='bold')
            ax.axis('off')
            figs.append((cluster_id, fig))

        return figs

    def reduce_dimensionality(self, X, method='umap', n_components=2):
        """Уменьшение размерности для визуализации"""
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            embedding = reducer.fit_transform(X_dense)

        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(X_dense) - 1)
            )
            embedding = reducer.fit_transform(X_dense)

        else:  # PCA по умолчанию
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dense)
            reducer = PCA(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(X_scaled)

        return embedding


def main():
    st.set_page_config(
        page_title="Text Clustering Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 Анализ кластеризации текстов")
    st.markdown("---")

    # Инициализация приложения
    if 'app' not in st.session_state:
        st.session_state.app = TextClusterApp()

    app = st.session_state.app

    # Боковая панель для настроек
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Загрузка данных
        st.subheader("📁 Данные")
        data_source = st.radio("Источник данных:", ["Пример данных", "Загрузить файл"])

        if data_source == "Пример данных":
            app.texts = app.load_sample_data()
            st.success(f"Загружено {len(app.texts)} примеров текстов")
        else:
            uploaded_file = st.file_uploader("Загрузите CSV или TXT файл", type=['csv', 'txt'])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    text_column = st.selectbox("Выберите колонку с текстом", df.columns)
                    app.texts = df[text_column].dropna().tolist()
                else:
                    content = uploaded_file.read().decode('utf-8')
                    app.texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.success(f"Загружено {len(app.texts)} текстов")

        # Настройки векторизации
        st.subheader("🔤 Векторизация")
        vectorization_method = st.selectbox(
            "Метод векторизации:",
            ['tfidf', 'count']
        )

        max_features = st.slider("Максимальное количество признаков:", 100, 5000, 2000)
        ngram_range = st.selectbox(
            "N-gram диапазон:",
            [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
            format_func=lambda x: f"{x[0]}-{x[1]}"
        )

        # Выбор алгоритма
        st.subheader("🎯 Алгоритм кластеризации")
        algorithm = st.selectbox(
            "Алгоритм:",
            ['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'HDBSCAN',
             'GaussianMixture', 'SpectralClustering', 'AgglomerativeClustering']
        )

        # Параметры алгоритмов
        st.subheader("📊 Параметры алгоритма")

        if algorithm in ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'AgglomerativeClustering']:
            n_clusters = st.slider("Количество кластеров:", 2, 20, 5)

        if algorithm == 'GaussianMixture':
            n_components = st.slider("Количество компонент:", 2, 20, 5)

        if algorithm == 'DBSCAN':
            eps = st.slider("EPS:", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples:", 2, 20, 5)

        if algorithm == 'HDBSCAN' and HDBSCAN_AVAILABLE:
            min_cluster_size = st.slider("Min Cluster Size:", 2, 50, 10)
            min_samples = st.slider("Min Samples:", 1, 20, 5)

        if algorithm == 'SpectralClustering':
            affinity = st.selectbox("Affinity:", ['rbf', 'nearest_neighbors', 'cosine'])
            if affinity == 'nearest_neighbors':
                n_neighbors = st.slider("N Neighbors:", 5, 50, 10)

        if algorithm == 'AgglomerativeClustering':
            linkage = st.selectbox("Linkage:", ['ward', 'complete', 'average', 'single'])
            if linkage == 'ward':
                st.info("Ward linkage требует евклидовой метрики")

        # Метод визуализации
        st.subheader("📈 Визуализация")
        viz_method = st.selectbox(
            "Метод уменьшения размерности:",
            ['umap', 'tsne', 'pca']
        )

    # Основная область
    if not app.texts:
        st.warning("⚠️ Пожалуйста, загрузите данные или используйте примеры")
        return

    # Векторизация
    with st.spinner("Векторизация текстов..."):
        X = app.vectorize_texts(
            app.texts,
            method=vectorization_method,
            max_features=max_features,
            ngram_range=ngram_range
        )

    # Подготовка параметров алгоритма
    algorithm_params = {}

    # Устанавливаем параметры в зависимости от алгоритма
    if algorithm in ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'AgglomerativeClustering']:
        algorithm_params['n_clusters'] = n_clusters
        if algorithm != 'AgglomerativeClustering':  # AgglomerativeClustering не поддерживает random_state
            algorithm_params['random_state'] = 42

    elif algorithm == 'GaussianMixture':
        algorithm_params['n_components'] = n_components
        algorithm_params['random_state'] = 42

    elif algorithm == 'DBSCAN':
        algorithm_params['eps'] = eps
        algorithm_params['min_samples'] = min_samples

    elif algorithm == 'HDBSCAN' and HDBSCAN_AVAILABLE:
        algorithm_params['min_cluster_size'] = min_cluster_size
        algorithm_params['min_samples'] = min_samples

    if algorithm == 'SpectralClustering':
        algorithm_params['affinity'] = affinity
        if affinity == 'nearest_neighbors':
            algorithm_params['n_neighbors'] = n_neighbors

    if algorithm == 'AgglomerativeClustering':
        algorithm_params['linkage'] = linkage

    # Кластеризация
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎯 Результаты кластеризации")

        if st.button("Запустить кластеризацию", type="primary"):
            with st.spinner("Выполняется кластеризация..."):
                try:
                    labels, model, exec_time = app.perform_clustering(algorithm, X, **algorithm_params)

                    # Вычисление метрик
                    metrics = app.calculate_metrics(X, labels)

                    # Отображение результатов
                    st.success(f"Кластеризация завершена за {exec_time:.2f} секунд")

                    # Метрики
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

                    with col_metric1:
                        st.metric("Кластеров", metrics['n_clusters'])
                    with col_metric2:
                        st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
                    with col_metric3:
                        st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}")
                    with col_metric4:
                        st.metric("Шумовые точки", metrics['n_noise'])

                    # Визуализация
                    st.subheader("📊 Визуализация кластеров")

                    with st.spinner("Строим визуализацию..."):
                        embedding = app.reduce_dimensionality(X, method=viz_method)

                        # Создаем DataFrame для визуализации
                        viz_df = pd.DataFrame({
                            'x': embedding[:, 0],
                            'y': embedding[:, 1],
                            'cluster': labels,
                            'text': app.texts
                        })

                        # Plotly scatter plot
                        fig = px.scatter(
                            viz_df,
                            x='x',
                            y='y',
                            color='cluster',
                            hover_data=['text'],
                            title=f'{algorithm} Clustering - {viz_method.upper()} проекция',
                            color_continuous_scale='viridis'
                        )

                        fig.update_traces(
                            marker=dict(size=8, opacity=0.7),
                            selector=dict(mode='markers')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Ключевые слова кластеров
                    st.subheader("🔤 Ключевые слова кластеров")

                    cluster_keywords = app.get_cluster_keywords(labels, n_words=10)

                    for cluster_id, keywords in cluster_keywords.items():
                        with st.expander(
                                f"Кластер {cluster_id} ({len([x for x in labels if x == cluster_id])} документов)"):
                            if keywords:
                                keywords_df = pd.DataFrame(keywords, columns=['Слово', 'Вес'])
                                st.dataframe(
                                    keywords_df.style.format({'Вес': '{:.4f}'}),
                                    use_container_width=True
                                )
                            else:
                                st.info("Нет ключевых слов")

                    # Облака слов
                    st.subheader("☁️ Облака слов кластеров")
                    wordcloud_figs = app.create_wordclouds(labels)

                    if wordcloud_figs:
                        cols = st.columns(2)
                        for idx, (cluster_id, fig) in enumerate(wordcloud_figs):
                            with cols[idx % 2]:
                                st.pyplot(fig)

                    # Детализация кластеров
                    st.subheader("📋 Детализация кластеров")

                    # Включаем шум (-1) в список кластеров
                    all_clusters = sorted(np.unique(labels))
                    selected_cluster = st.selectbox(
                        "Выберите кластер для детального просмотра:",
                        all_clusters,
                        format_func=lambda x: f"Кластер {x}" if x != -1 else "Шум (-1)"
                    )

                    cluster_texts = [text for i, text in enumerate(app.texts) if labels[i] == selected_cluster]

                    if cluster_texts:
                        st.write(f"Документов в кластере {selected_cluster}: {len(cluster_texts)}")

                        for i, text in enumerate(cluster_texts[:10]):  # Показываем первые 10
                            with st.expander(f"Документ {i + 1}"):
                                st.write(text)

                    # Сохранение результатов
                    st.subheader("💾 Сохранение результатов")

                    results_df = pd.DataFrame({
                        'text': app.texts,
                        'cluster': labels
                    })

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Скачать результаты CSV",
                        data=csv,
                        file_name=f"clustering_results_{algorithm}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Ошибка при кластеризации: {str(e)}")
                    st.error("Подробности ошибки:")
                    st.code(str(e))

    with col2:
        st.header("ℹ️ Информация")

        st.subheader("О алгоритмах:")

        algorithm_info = {
            'KMeans': "• Разделительная кластеризация\n• Требует указания k\n• Чувствителен к выбросам",
            'MiniBatchKMeans': "• Быстрая версия KMeans\n• Подходит для больших данных\n• Приближенное решение",
            'DBSCAN': "• Плотностная кластеризация\n• Обнаруживает шум\n• Не требует указания k",
            'HDBSCAN': "• Иерархическая DBSCAN\n• Автоматический выбор кластеров\n• Устойчив к шуму",
            'GaussianMixture': "• Вероятностная модель\n• Мягкая кластеризация\n• Использует n_components вместо n_clusters",
            'SpectralClustering': "• На основе спектра графов\n• Работает с невыпуклыми кластерами\n• Вычислительно сложный",
            'AgglomerativeClustering': "• Иерархическая кластеризация\n• Построение дендрограмм\n• Разные стратегии связи"
        }

        st.info(algorithm_info.get(algorithm, "Выберите алгоритм для получения информации"))

        st.subheader("Метрики качества:")
        st.markdown("""
        - **Silhouette**: [-1, 1] - чем выше, тем лучше
        - **Calinski-Harabasz**: [0, ∞] - чем выше, тем лучше  
        - **Davies-Bouldin**: [0, ∞] - чем ниже, тем лучше
        """)

        if not UMAP_AVAILABLE:
            st.warning("""
            ⚠️ UMAP не установлен. 
            Установите: `pip install umap-learn`
            """)

        if not HDBSCAN_AVAILABLE and algorithm == 'HDBSCAN':
            st.error("""
            ❌ HDBSCAN не установлен.
            Установите: `pip install hdbscan`
            """)


if __name__ == "__main__":
    main()