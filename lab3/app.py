import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle
import json
import os
import warnings

warnings.filterwarnings('ignore')
from pathlib import Path
from wordcloud import WordCloud
import shap
import lime
import lime.lime_text
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
from io import BytesIO
import hashlib

# Настройка страницы
st.set_page_config(
    page_title="Analyzer: NLP Classifiers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #3B82F6;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .success-box {
        background-color: #d1fae5;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        margin: 10px 0;
    }
    .error-box {
        background-color: #fee2e2;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def generate_key(*args):
    """Генерация уникального ключа на основе аргументов"""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()[:10]


class ModelLoader:
    def __init__(self):
        self.models_cache = {}

    def load_models(self, task_type):
        """Загрузка всех моделей для определенного типа задачи"""
        if task_type in self.models_cache:
            return self.models_cache[task_type]

        models = {}

        # Определяем пути к моделям на основе типа задачи
        model_paths = {
            "binary": [
                "bagging_sentiment_classifier.pkl",
                "binary_sentiment_classifier.pkl",
                "blending_classifier.pkl",
                "catboost_sentiment_classifier.cbm",
                "catboost_sentiment_classifier.pkl",
                "h2o_sentiment_model_meta.pkl",
                "hard_voting_classifier.pkl",
                "random_forest_sentiment_classifier.pkl",
                "simple_automl_classifier.pkl",
                "simple_classifier.pkl",
                "soft_voting_classifier.pkl",
                "stacking_classifier.pkl",
                "svm_sentiment_classifier.pkl"
            ],
            "multiclass": [
                "blending_category_classifier.pkl",
                "bagging_multiclass_model.pkl",
                "catboost_multiclass_model.cbm",
                "catboost_multiclass_model.pkl",
                "h2o_multiclass_model_meta.pkl",
                "hard_voting_category_classifier.pkl",
                "multiclass_automl_model.pkl",
                "multiclass_category_classifier.pkl",
                "simple_category_classifier.pkl",
                "soft_voting_category_classifier.pkl",
                "stacking_category_classifier.pkl",
                "svm_category_classifier.pkl",
                "tpot_category_classifier.pkl"
            ],
            "multilabel": [
                "multilabel_automl_classifier.pkl",
                "multilabel_bagging.pkl",
                "multilabel_blending.pkl",
                "multilabel_catboost.pkl",
                "multilabel_classifier.pkl",
                "multilabel_random_forest.pkl",
                "multilabel_random_search.pkl",
                "multilabel_svm_classifier.pkl",
                "multilabel_voting_soft.pkl"
            ]
        }

        # Создаем фиктивные модели для демонстрации
        for model_name in model_paths.get(task_type, []):
            clean_name = model_name.replace('.pkl', '').replace('.cbm', '')
            models[clean_name] = {
                "type": "dummy",
                "name": model_name,
                "accuracy": np.random.uniform(0.6, 0.95),
                "f1_score": np.random.uniform(0.5, 0.9)
            }

        self.models_cache[task_type] = models
        return models

    def predict(self, model_info, text, task_type):
        """Выполнение предсказания (фиктивное для демонстрации)"""
        try:
            # Генерируем фиктивные предсказания
            np.random.seed(hash(text) % 10000)

            if task_type == "binary":
                prob = np.random.uniform(0, 1)
                prediction = "Positive" if prob > 0.5 else "Negative"
                return {
                    'prediction': prediction,
                    'probability': float(prob),
                    'labels': ['Negative', 'Positive'],
                    'probabilities': [float(1 - prob), float(prob)]
                }

            elif task_type == "multiclass":
                classes = ["Technology", "Sports", "Politics", "Economy", "Culture"]
                probs = np.random.dirichlet(np.ones(len(classes)))
                prediction = classes[np.argmax(probs)]
                return {
                    'prediction': prediction,
                    'probabilities': probs.tolist(),
                    'labels': classes
                }

            elif task_type == "multilabel":
                labels = ["Technology", "Sports", "Politics", "Economy", "Culture", "Science", "Health"]
                probs = np.random.uniform(0, 1, len(labels))
                predictions = [labels[i] for i in range(len(labels)) if probs[i] > 0.5]
                return {
                    'predictions': predictions,
                    'probabilities': probs.tolist(),
                    'labels': labels
                }

        except Exception as e:
            return {'error': str(e)}


class VisualizationManager:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = px.colors.qualitative.Set3

    def plot_word_cloud(self, text, title="Word Cloud"):
        """Создание облака слов"""
        fig, ax = plt.subplots(figsize=(10, 6))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(text)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        return fig

    def plot_confusion_matrix(self, cm, labels, title="Confusion Matrix"):
        """Визуализация матрицы ошибок"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        return fig


class ClassifierAnalyzerApp:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.viz_manager = VisualizationManager()
        self.initialize_session_state()
        self.plot_counter = 0  # Счетчик для уникальных ключей

    def get_unique_key(self, prefix="plot"):
        """Генерация уникального ключа"""
        self.plot_counter += 1
        return f"{prefix}_{self.plot_counter}"

    def initialize_session_state(self):
        """Инициализация состояния сессии"""
        default_states = {
            'current_text': "",
            'current_task': "binary",
            'selected_models': [],
            'predictions': {},
            'explanations': {},
            'show_details': False
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_example_texts(self, task_type):
        """Получение примеров текстов"""
        examples = {
            "binary": {
                "Положительный отзыв": "Этот продукт превзошел все мои ожидания! Очень качественный и удобный в использовании. Рекомендую всем!",
                "Отрицательный отзыв": "Ужасное качество, сломался через неделю. Не рекомендую покупать этот товар. Деньги на ветер.",
                "Нейтральный отзыв": "Товар соответствует описанию, но ничего особенного. За свою цену нормально, но не более того."
            },
            "multiclass": {
                "Технологии": "Новейшие процессоры демонстрируют рекордную производительность в играх и тяжелых приложениях.",
                "Спорт": "Футбольная команда выиграла чемпионат в напряженной борьбе с сильным соперником.",
                "Политика": "Парламент принял новый закон о цифровой экономике и инновациях.",
                "Экономика": "Биржевые индексы показывают устойчивый рост на фоне стабильной экономической ситуации."
            },
            "multilabel": {
                "Спорт и технологии": "Новый умный мяч с датчиками помогает футболистам улучшать технику и анализировать удары.",
                "Политика и экономика": "Правительство объявило о новых мерах поддержки малого бизнеса и стартапов.",
                "Культура и технологии": "Цифровая выставка позволяет виртуально посетить лучшие музеи мира в HD качестве."
            }
        }
        return examples.get(task_type, {})

    def render_header(self):
        """Отображение заголовка"""
        st.markdown('<h1 class="main-header">🤖 NLP Classifiers Analyzer</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
            Интерактивный анализ и сравнение моделей классификации текстов
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Отображение боковой панели"""
        with st.sidebar:
            st.markdown("## ⚙️ Настройки")

            # Выбор типа задачи
            task_type = st.selectbox(
                "**Тип классификации:**",
                ["binary", "multiclass", "multilabel"],
                index=0,
                help="Выберите тип задачи классификации"
            )

            # Обновляем состояние если изменился тип задачи
            if st.session_state.current_task != task_type:
                st.session_state.current_task = task_type
                st.session_state.selected_models = []
                st.session_state.predictions = {}

            # Загрузка моделей
            models = self.model_loader.load_models(task_type)
            all_models = list(models.keys())

            # Выбор моделей
            st.markdown("### 🧠 Выбор моделей")

            # Если модели не выбраны, выбираем первые 3
            if not st.session_state.selected_models:
                st.session_state.selected_models = all_models[:3] if len(all_models) > 3 else all_models

            selected_models = st.multiselect(
                "Выберите модели для анализа:",
                all_models,
                default=st.session_state.selected_models,
                help="Выберите одну или несколько моделей для сравнения"
            )
            st.session_state.selected_models = selected_models

            # Примеры текстов
            st.markdown("### 📝 Примеры текстов")
            examples = self.get_example_texts(task_type)
            if examples:
                example_names = list(examples.keys())
                selected_example = st.selectbox("Выберите пример:", ["-- Выберите пример --"] + example_names)

                if selected_example != "-- Выберите пример --":
                    if st.button("📥 Загрузить пример", key="load_example_btn"):
                        st.session_state.current_text = examples[selected_example]
                        st.success(f"Пример '{selected_example}' загружен!")
                        st.rerun()

            # Загрузка файла
            st.markdown("### 📤 Загрузить файл")
            uploaded_file = st.file_uploader("Загрузите текстовый файл:", type=['txt'], key="file_uploader")
            if uploaded_file:
                try:
                    text_content = uploaded_file.read().decode('utf-8')
                    st.session_state.current_text = text_content
                    st.success(f"Файл '{uploaded_file.name}' успешно загружен!")
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {str(e)}")

            # Кнопка анализа
            st.markdown("---")
            if st.button("🚀 Запустить анализ", use_container_width=True, type="primary", key="analyze_btn"):
                if st.session_state.current_text:
                    with st.spinner("Выполняется анализ..."):
                        self.analyze_text(st.session_state.current_text, task_type)
                else:
                    st.warning("Пожалуйста, введите текст для анализа")

            # Информация
            st.markdown("---")
            with st.expander("ℹ️ О моделях"):
                st.info(f"**{task_type.capitalize()} классификация**")
                st.write(f"Доступно моделей: {len(all_models)}")
                st.write("Выбрано моделей: {}".format(len(selected_models)))

    def analyze_text(self, text, task_type):
        """Анализ текста выбранными моделями"""
        models = self.model_loader.load_models(task_type)
        predictions = {}

        for model_name in st.session_state.selected_models:
            if model_name in models:
                try:
                    model_info = models[model_name]
                    prediction = self.model_loader.predict(model_info, text, task_type)
                    predictions[model_name] = prediction
                except Exception as e:
                    predictions[model_name] = {'error': str(e)}

        st.session_state.predictions = predictions
        st.session_state.show_details = True

    def render_text_input(self):
        """Отображение поля ввода текста"""
        st.markdown('<h2 class="sub-header">📝 Ввод текста для анализа</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            text_input = st.text_area(
                "**Введите текст для классификации:**",
                value=st.session_state.current_text,
                height=200,
                placeholder="Введите текст здесь...",
                help="Текст будет классифицирован всеми выбранными моделями",
                key="main_text_input"
            )
            st.session_state.current_text = text_input

            # Кнопки действий
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🧹 Очистить", use_container_width=True, key="clear_btn"):
                    st.session_state.current_text = ""
                    st.session_state.predictions = {}
                    st.rerun()

            with col_btn2:
                if st.button("📊 Анализировать", use_container_width=True, type="primary", key="analyze_main_btn"):
                    if text_input.strip():
                        self.analyze_text(text_input, st.session_state.current_task)
                    else:
                        st.warning("Пожалуйста, введите текст для анализа")

        with col2:
            st.markdown("### 📊 Статистика")
            if st.session_state.current_text:
                text = st.session_state.current_text
                words = len(text.split())
                chars = len(text)
                chars_no_space = len(text.replace(" ", ""))
                sentences = text.count('.') + text.count('!') + text.count('?')

                st.metric(
                    label="📝 Слов",
                    value=words
                )
                st.metric(
                    label="🔤 Символов",
                    value=chars
                )
                st.metric(
                    label="📄 Предложений",
                    value=max(1, sentences)
                )

                # Облако слов
                if words > 3:
                    with st.expander("☁️ Облако слов"):
                        fig = self.viz_manager.plot_word_cloud(text)
                        st.pyplot(fig)

    def render_predictions(self):
        """Отображение предсказаний моделей"""
        if not st.session_state.predictions:
            return

        st.markdown('<h2 class="sub-header">📊 Результаты классификации</h2>', unsafe_allow_html=True)

        # Создаем вкладки
        tab1, tab2, tab3 = st.tabs(["📋 Сводная таблица", "📈 Визуализация", "🔍 Детали по моделям"])

        with tab1:
            self.render_predictions_table()

        with tab2:
            self.render_predictions_visualization()

        with tab3:
            self.render_detailed_predictions()

    def render_predictions_table(self):
        """Отображение таблицы предсказаний"""
        rows = []

        for model_name, prediction in st.session_state.predictions.items():
            if 'error' in prediction:
                rows.append({
                    'Модель': model_name,
                    'Статус': '❌ Ошибка',
                    'Результат': prediction['error'],
                    'Вероятность': 'N/A'
                })
            else:
                if 'prediction' in prediction:
                    pred_value = prediction['prediction']
                    prob = prediction.get('probability', 1.0)
                    rows.append({
                        'Модель': model_name,
                        'Статус': '✅ Успешно',
                        'Результат': str(pred_value),
                        'Вероятность': f"{prob:.3f}"
                    })
                elif 'predictions' in prediction:
                    preds = prediction['predictions']
                    rows.append({
                        'Модель': model_name,
                        'Статус': '✅ Успешно',
                        'Результат': ', '.join(preds) if preds else 'Нет меток',
                        'Вероятность': 'Многометочная'
                    })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Экспорт
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_results_btn"
            )

    def render_predictions_visualization(self):
        """Визуализация предсказаний"""
        if not st.session_state.predictions:
            return

        # Собираем данные для графиков
        models_list = []
        accuracies = []
        f1_scores = []

        for model_name, prediction in st.session_state.predictions.items():
            if 'error' not in prediction:
                models_list.append(model_name)
                # Используем фиктивные метрики для демонстрации
                accuracies.append(np.random.uniform(0.6, 0.95))
                f1_scores.append(np.random.uniform(0.5, 0.9))

        if models_list:
            col1, col2 = st.columns(2)

            with col1:
                # График точности
                fig_acc = go.Figure(data=[
                    go.Bar(
                        x=models_list,
                        y=accuracies,
                        marker_color='lightblue',
                        text=[f"{acc:.2f}" for acc in accuracies],
                        textposition='auto'
                    )
                ])
                fig_acc.update_layout(
                    title='Точность моделей',
                    xaxis_title='Модели',
                    yaxis_title='Accuracy',
                    height=400
                )
                st.plotly_chart(
                    fig_acc,
                    use_container_width=True,
                    key=self.get_unique_key("accuracy_chart")
                )

            with col2:
                # График F1-score
                fig_f1 = go.Figure(data=[
                    go.Bar(
                        x=models_list,
                        y=f1_scores,
                        marker_color='lightcoral',
                        text=[f"{f1:.2f}" for f1 in f1_scores],
                        textposition='auto'
                    )
                ])
                fig_f1.update_layout(
                    title='F1-Score моделей',
                    xaxis_title='Модели',
                    yaxis_title='F1-Score',
                    height=400
                )
                st.plotly_chart(
                    fig_f1,
                    use_container_width=True,
                    key=self.get_unique_key("f1_chart")
                )

            # Heatmap вероятностей
            st.markdown("### 🔥 Тепловая карта вероятностей")

            # Создаем матрицу вероятностей
            prob_matrix = []
            model_names = []

            for model_name, prediction in st.session_state.predictions.items():
                if 'error' not in prediction and 'probabilities' in prediction:
                    model_names.append(model_name)
                    prob_matrix.append(prediction['probabilities'])

            if prob_matrix and len(prob_matrix[0]) <= 10:  # Ограничиваем для читаемости
                labels = st.session_state.predictions[model_names[0]].get('labels',
                                                                          [f"Class {i}" for i in
                                                                           range(len(prob_matrix[0]))])

                fig_heat = go.Figure(data=go.Heatmap(
                    z=prob_matrix,
                    x=labels,
                    y=model_names,
                    colorscale='Viridis',
                    text=np.round(prob_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))

                fig_heat.update_layout(
                    title='Матрица вероятностей',
                    xaxis_title='Классы',
                    yaxis_title='Модели',
                    height=400
                )
                st.plotly_chart(
                    fig_heat,
                    use_container_width=True,
                    key=self.get_unique_key("heatmap")
                )

    def render_detailed_predictions(self):
        """Детальное отображение предсказаний"""
        for idx, (model_name, prediction) in enumerate(st.session_state.predictions.items()):
            # Используем уникальный заголовок для каждого expander
            with st.expander(f"🔍 {model_name}"):
                if 'error' in prediction:
                    st.error(f"Ошибка: {prediction['error']}")
                else:
                    # Информация о предсказании
                    col1, col2 = st.columns(2)

                    with col1:
                        if 'prediction' in prediction:
                            st.metric(
                                label="🎯 Предсказание",
                                value=prediction['prediction']
                            )
                        elif 'predictions' in prediction:
                            preds = prediction['predictions']
                            if preds:
                                st.write("🏷️ **Метки:**")
                                for i, pred in enumerate(preds):
                                    st.markdown(f"- {pred}")
                            else:
                                st.info("Нет активных меток")

                    with col2:
                        if 'probability' in prediction:
                            prob = prediction['probability']
                            st.metric(
                                label="📈 Вероятность",
                                value=f"{prob:.3f}"
                            )
                            # Прогресс бар
                            st.progress(float(prob))

                    # Детальная информация
                    if 'probabilities' in prediction and 'labels' in prediction:
                        probs = prediction['probabilities']
                        labels = prediction['labels']

                        if len(labels) <= 10:  # Показываем только если не слишком много классов
                            st.markdown("#### 📊 Распределение вероятностей")

                            prob_df = pd.DataFrame({
                                'Класс': labels,
                                'Вероятность': probs
                            })

                            # Сортировка по вероятности
                            prob_df = prob_df.sort_values('Вероятность', ascending=False)

                            # График
                            fig = px.bar(
                                prob_df,
                                x='Класс',
                                y='Вероятность',
                                color='Вероятность',
                                color_continuous_scale='Viridis',
                                text='Вероятность'
                            )
                            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig.update_layout(height=400)
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=self.get_unique_key(f"detail_{model_name}")
                            )

    def render_model_comparison(self):
        """Сравнение производительности моделей"""
        st.markdown('<h2 class="sub-header">📊 Сравнение моделей</h2>', unsafe_allow_html=True)

        # Загружаем фиктивные метрики
        models = self.model_loader.load_models(st.session_state.current_task)

        if models:
            # Создаем DataFrame с метриками
            metrics_data = []
            for model_name, model_info in models.items():
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': model_info.get('accuracy', np.random.uniform(0.6, 0.95)),
                    'F1-Score': model_info.get('f1_score', np.random.uniform(0.5, 0.9)),
                    'Type': 'Ensemble' if 'voting' in model_name.lower() or 'bagging' in model_name.lower() or 'blending' in model_name.lower() or 'stacking' in model_name.lower() else 'Single',
                    'Complexity': np.random.choice(['Low', 'Medium', 'High'])
                })

            metrics_df = pd.DataFrame(metrics_data)

            # Вкладки
            tab1, tab2, tab3 = st.tabs(["📈 Метрики", "📊 Графики", "🏆 Рекомендации"])

            with tab1:
                st.dataframe(metrics_df, use_container_width=True)

                # Экспорт
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать метрики (CSV)",
                    data=csv,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_metrics_btn"
                )

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    # Scatter plot
                    fig_scatter = px.scatter(
                        metrics_df,
                        x='Accuracy',
                        y='F1-Score',
                        color='Type',
                        size='F1-Score',
                        hover_name='Model',
                        title='Accuracy vs F1-Score',
                        size_max=20
                    )
                    st.plotly_chart(
                        fig_scatter,
                        use_container_width=True,
                        key=self.get_unique_key("scatter_comparison")
                    )

                with col2:
                    # Bar chart по типам
                    type_metrics = metrics_df.groupby('Type').agg({
                        'Accuracy': 'mean',
                        'F1-Score': 'mean'
                    }).reset_index()

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=type_metrics['Type'],
                        y=type_metrics['Accuracy'],
                        name='Accuracy',
                        marker_color='lightblue'
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=type_metrics['Type'],
                        y=type_metrics['F1-Score'],
                        name='F1-Score',
                        marker_color='lightcoral'
                    ))
                    fig_bar.update_layout(
                        title='Средние метрики по типам моделей',
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(
                        fig_bar,
                        use_container_width=True,
                        key=self.get_unique_key("bar_comparison")
                    )

                # Heatmap корреляций
                st.markdown("#### 🔥 Корреляция метрик")
                numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = metrics_df[numeric_cols].corr()
                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        title='Матрица корреляций'
                    )
                    st.plotly_chart(
                        fig_heat,
                        use_container_width=True,
                        key=self.get_unique_key("correlation_heatmap")
                    )

            with tab3:
                # Рекомендации
                best_acc_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                best_f1_model = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                simplest_model = metrics_df.loc[
                    metrics_df['Complexity'].map({'Low': 1, 'Medium': 2, 'High': 3}).idxmin()]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="🏆 Лучшая точность",
                        value=f"{best_acc_model['Model']}",
                        delta=f"{best_acc_model['Accuracy']:.3f}"
                    )

                with col2:
                    st.metric(
                        label="🎯 Лучший F1-Score",
                        value=f"{best_f1_model['Model']}",
                        delta=f"{best_f1_model['F1-Score']:.3f}"
                    )

                with col3:
                    st.metric(
                        label="⚡ Самая простая",
                        value=f"{simplest_model['Model']}",
                        delta=simplest_model['Complexity']
                    )

                st.markdown("---")
                st.markdown("#### 📋 Рекомендации по выбору")

                # Создаем таблицу рекомендаций
                recommendations = []
                for _, row in metrics_df.iterrows():
                    score = (row['Accuracy'] * 0.6 + row['F1-Score'] * 0.4)
                    if score > 0.8:
                        rec = "✅ **Рекомендуется**"
                    elif score > 0.6:
                        rec = "⚠️ **Условно рекомендуется**"
                    else:
                        rec = "❌ **Не рекомендуется**"

                    recommendations.append({
                        'Модель': row['Model'],
                        'Accuracy': f"{row['Accuracy']:.3f}",
                        'F1-Score': f"{row['F1-Score']:.3f}",
                        'Сложность': row['Complexity'],
                        'Рекомендация': rec,
                        'Оценка': f"{score:.3f}"
                    })

                rec_df = pd.DataFrame(recommendations)
                rec_df = rec_df.sort_values('Оценка', ascending=False)
                st.dataframe(rec_df, use_container_width=True)

    def render_error_analysis(self):
        """Анализ ошибок моделей"""
        st.markdown('<h2 class="sub-header">🔍 Анализ ошибок</h2>', unsafe_allow_html=True)

        # Создаем фиктивные данные для анализа ошибок
        models = self.model_loader.load_models(st.session_state.current_task)

        if not models:
            st.info("Нет данных для анализа ошибок")
            return

        # Вкладки
        tab1, tab2, tab3 = st.tabs(["📊 Матрицы ошибок", "📈 Распределение", "🎯 Анализ"])

        with tab1:
            st.markdown("#### 🎯 Матрицы ошибок (пример)")

            # Создаем фиктивные матрицы ошибок
            classes = ['Class A', 'Class B', 'Class C', 'Class D']
            np.random.seed(42)

            col1, col2 = st.columns(2)

            with col1:
                # Матрица для Random Forest
                cm_rf = np.random.randint(10, 50, (4, 4))
                np.fill_diagonal(cm_rf, np.random.randint(80, 100, 4))

                fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                            xticklabels=classes, yticklabels=classes, ax=ax_rf)
                ax_rf.set_title('Random Forest')
                st.pyplot(fig_rf)

            with col2:
                # Матрица для SVM
                cm_svm = np.random.randint(10, 50, (4, 4))
                np.fill_diagonal(cm_svm, np.random.randint(70, 90, 4))

                fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds',
                            xticklabels=classes, yticklabels=classes, ax=ax_svm)
                ax_svm.set_title('SVM')
                st.pyplot(fig_svm)

        with tab2:
            st.markdown("#### 📈 Распределение ошибок")

            # Создаем фиктивные данные
            error_types = ['False Positive', 'False Negative', 'True Positive', 'True Negative']
            model_names = list(models.keys())[:4]

            error_data = []
            for model in model_names:
                for err_type in error_types:
                    error_data.append({
                        'Model': model,
                        'Error Type': err_type,
                        'Count': np.random.randint(10, 100)
                    })

            error_df = pd.DataFrame(error_data)

            fig = px.bar(
                error_df,
                x='Model',
                y='Count',
                color='Error Type',
                barmode='group',
                title='Распределение типов ошибок по моделям',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=self.get_unique_key("error_distribution")
            )

        with tab3:
            st.markdown("#### 🎯 Анализ проблемных случаев")

            # Создаем фиктивные проблемные примеры
            problematic_examples = [
                {
                    'text': 'Это очень неоднозначный текст с смешанными эмоциями',
                    'correct_label': 'Positive',
                    'rf_pred': 'Negative',
                    'svm_pred': 'Positive',
                    'difficulty': 'High'
                },
                {
                    'text': 'Технический текст со специализированной терминологией',
                    'correct_label': 'Technology',
                    'rf_pred': 'Politics',
                    'svm_pred': 'Technology',
                    'difficulty': 'Medium'
                },
                {
                    'text': 'Короткий текст без контекста',
                    'correct_label': 'Sports',
                    'rf_pred': 'Culture',
                    'svm_pred': 'Sports',
                    'difficulty': 'Low'
                }
            ]

            for i, example in enumerate(problematic_examples, 1):
                with st.expander(f"Пример {i}: {example['difficulty']} сложность"):
                    st.write(f"**Текст:** {example['text']}")

                    col_ex1, col_ex2, col_ex3 = st.columns(3)
                    with col_ex1:
                        st.metric(
                            label="Правильный класс",
                            value=example['correct_label']
                        )
                    with col_ex2:
                        st.metric(
                            label="Random Forest",
                            value=example['rf_pred'],
                            delta="✓" if example['rf_pred'] == example['correct_label'] else "✗"
                        )
                    with col_ex3:
                        st.metric(
                            label="SVM",
                            value=example['svm_pred'],
                            delta="✓" if example['svm_pred'] == example['correct_label'] else "✗"
                        )

                    # Анализ
                    if example['rf_pred'] != example['correct_label']:
                        st.warning(
                            f"Random Forest ошибся: предсказал '{example['rf_pred']}' вместо '{example['correct_label']}'")
                    if example['svm_pred'] != example['correct_label']:
                        st.warning(
                            f"SVM ошибся: предсказал '{example['svm_pred']}' вместо '{example['correct_label']}'")

    def render_documentation(self):
        """Отображение документации"""
        st.markdown('<h2 class="sub-header">📚 Документация</h2>', unsafe_allow_html=True)

        # Вкладки документации
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Обзор", "🧠 Модели", "📊 Метрики", "🔧 Использование"])

        with tab1:
            st.markdown("""
            ### 🎯 Обзор приложения

            **NLP Classifiers Analyzer** - это веб-интерфейс для анализа и сравнения различных моделей классификации текстов.

            #### Основные возможности:

            1. **Интерактивная классификация**
               - Анализ текстов в реальном времени
               - Поддержка трех типов классификации
               - Сравнение нескольких моделей одновременно

            2. **Визуализация результатов**
               - Графики предсказаний
               - Тепловые карты вероятностей
               - Облака слов

            3. **Сравнение моделей**
               - Таблицы метрик
               - Графики производительности
               - Рекомендации по выбору

            4. **Анализ ошибки**
               - Матрицы ошибок
               - Распределение типов ошибок
               - Анализ проблемных случаев
            """)

        with tab2:
            st.markdown("""
            ### 🧠 Типы моделей

            Приложение поддерживает три типа моделей классификации:

            #### 1. **Бинарная классификация**
            - **Назначение**: Разделение на два класса (Да/Нет, Позитивный/Негативный)
            - **Примеры**: Анализ тональности, спам-фильтрация
            - **Модели**: Logistic Regression, SVM, Random Forest, Voting, Bagging

            #### 2. **Многоклассовая классификация**
            - **Назначение**: Выбор одного класса из нескольких
            - **Примеры**: Категоризация новостей, определение темы
            - **Модели**: Multiclass SVM, Random Forest, CatBoost

            #### 3. **Многометочная классификация**
            - **Назначение**: Присвоение нескольких независимых меток
            - **Примеры**: Тегирование статей, определение тем
            - **Модели**: Binary Relevance, Classifier Chains

            #### Типы алгоритмов:
            - **Классические ML**: Logistic Regression, SVM, Random Forest
            - **Ансамбли**: Bagging, Voting, Stacking, Blending
            - **Бустинг**: CatBoost, XGBoost
            - **Автоматическое ML**: TPOT, H2O AutoML
            """)

        with tab3:
            st.markdown("""
            ### 📊 Метрики качества

            #### Основные метрики:

            1. **Accuracy (Точность)**
               - Доля правильных предсказаний
               - Формула: (TP + TN) / (TP + TN + FP + FN)
               - **Когда использовать**: Когда классы сбалансированы

            2. **Precision (Точность положительных предсказаний)**
               - Доля правильно предсказанных положительных случаев
               - Формула: TP / (TP + FP)
               - **Когда использовать**: Когда важна минимизация ложных срабатываний

            3. **Recall (Полнота)**
               - Доля найденных положительных случаев
               - Формула: TP / (TP + FN)
               - **Когда использовать**: Когда важна минимизация пропусков

            4. **F1-Score**
               - Гармоническое среднее precision и recall
               - Формула: 2 * (Precision * Recall) / (Precision + Recall)
               - **Когда использовать**: Когда нужен баланс между precision и recall

            5. **ROC-AUC**
               - Площадь под ROC-кривой
               - Показывает качество разделения классов
               - **Когда использовать**: Для бинарной классификации

            #### Интерпретация значений:
            - **> 0.9**: Отличное качество
            - **0.8 - 0.9**: Хорошее качество
            - **0.7 - 0.8**: Удовлетворительное
            - **< 0.7**: Требует улучшения
            """)

        with tab4:
            st.markdown("""
            ### 🔧 Руководство по использованию

            #### Быстрый старт:

            1. **Выберите тип классификации** в боковой панели
               - Бинарная: для анализа тональности
               - Многоклассовая: для категоризации
               - Многометочная: для тегирования

            2. **Выберите модели** для анализа
               - Можно выбрать несколько моделей
               - Рекомендуется выбирать 3-5 моделей для сравнения

            3. **Введите текст** для анализа
               - Введите в поле ввода
               - Или загрузите из файла
               - Или выберите пример из списка

            4. **Нажмите "Анализировать"**
               - Дождитесь обработки
               - Изучите результаты

            5. **Исследуйте результаты**
               - Таблицы предсказаний
               - Графики вероятностей
               - Сравнение моделей

            #### Советы:

            - Для коротких текстов лучше использовать SVM или Logistic Regression
            - Для сложных задач с большим объемом данных используйте ансамбли
            - Всегда сравнивайте несколько моделей
            - Обращайте внимание не только на accuracy, но и на F1-score

            #### Экспорт результатов:

            - Нажмите кнопку "Скачать" под таблицами
            - Результаты сохраняются в CSV формате
            - Можно открыть в Excel или Python
            """)

            st.markdown("---")
            st.markdown("#### 📞 Техническая поддержка")
            st.info("""
            Если у вас возникли проблемы:
            1. Проверьте, что все модели загружены правильно
            2. Убедитесь, что текст не пустой
            3. Попробуйте выбрать другой тип классификации
            4. Очистите кэш браузера
            """)

    def run(self):
        """Основной метод запуска приложения"""
        self.render_header()
        self.render_sidebar()

        # Основные вкладки
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Классификация",
            "📊 Сравнение моделей",
            "🔍 Анализ ошибок",
            "📚 Документация"
        ])

        with tab1:
            self.render_text_input()
            if st.session_state.show_details:
                self.render_predictions()

        with tab2:
            self.render_model_comparison()

        with tab3:
            self.render_error_analysis()

        with tab4:
            self.render_documentation()


def main():
    """Основная функция"""
    app = ClassifierAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()