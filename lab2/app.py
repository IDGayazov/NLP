import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec, Doc2Vec
import json
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_model():
    """Загрузка модели с обработкой ошибок"""
    try:
        # Пробуем загрузить как Doc2Vec модель
        model = Doc2Vec.load("doc2vec_pv-dm_20251023_181820.model")
        st.sidebar.success("✅ Модель Doc2Vec загружена")
        return model
    except:
        try:
            # Пробуем загрузить как Word2Vec модель
            model = Word2Vec.load("doc2vec_pv-dm_20251023_181820.model")
            st.sidebar.success("✅ Модель Word2Vec загружена")
            return model
        except:
            st.error("❌ Не удалось загрузить модель. Убедитесь что файлы моделей существуют:")
            st.code("""
Доступные форматы:
- doc2vec_pv-dm_20251023_181820.model (Doc2Vec)
- word2vec_cbow.model (Word2Vec)
- word2vec_skipgram.model (Word2Vec)
            """)
            return None

def load_model_metadata():
    """Загрузка метаданных модели"""
    try:
        with open("/home/ilnaz/code/NLP/lab2/doc2vec_pv-dm_20251023_181820_metadata.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"error": "Метаданные не найдены"}

def vector_arithmetic_interface(model):
    """Интерфейс векторной арифметики"""
    st.header("🔢 Векторная арифметика")
    
    if model is None:
        st.error("Модель не загружена")
        return
    
    expression = st.text_input("Введите выражение (например: казан - татарстан + россия):")
    
    if expression:
        try:
            words = expression.split()
            positives = []
            negatives = []
            
            # Парсим выражение
            i = 0
            while i < len(words):
                if words[i] == '+':
                    if i+1 < len(words):
                        positives.append(words[i+1])
                    i += 2
                elif words[i] == '-':
                    if i+1 < len(words):
                        negatives.append(words[i+1])
                    i += 2
                else:
                    positives.append(words[i])
                    i += 1
            
            # Проверяем наличие слов в модели
            missing_words = []
            for word in positives + negatives:
                if word not in model.wv:
                    missing_words.append(word)
            
            if missing_words:
                st.error(f"Слова не найдены в модели: {missing_words}")
                return
            
            # Выполняем векторную арифметику
            result = model.wv.most_similar(positive=positives, negative=negatives, topn=10)
            
            # Выводим результат
            st.subheader("Результат:")
            df = pd.DataFrame(result, columns=['Слово', 'Сходство'])
            st.dataframe(df)
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(10, 6))
            words_plot = [word for word, _ in result[:8]]
            similarities = [sim for _, sim in result[:8]]
            
            ax.barh(words_plot, similarities, color='skyblue')
            ax.set_xlabel('Косинусное сходство')
            ax.set_title('Топ-8 ближайших слов')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

def similarity_analysis_interface(model):
    """Анализ семантического сходства"""
    st.header("📊 Анализ сходства")
    
    if model is None:
        st.error("Модель не загружена")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        word1 = st.text_input("Слово 1:", "матбугат")
    with col2:
        word2 = st.text_input("Слово 2:", "хезмәте")
    
    if word1 and word2:
        try:
            # Проверяем наличие слов
            if word1 not in model.wv:
                st.error(f"Слово '{word1}' не найдено в модели")
                return
            if word2 not in model.wv:
                st.error(f"Слово '{word2}' не найдено в модели")
                return
            
            similarity = model.wv.similarity(word1, word2)
            st.metric("Косинусное сходство", f"{similarity:.4f}")
            
            # Ближайшие соседи для обоих слов
            neighbors1 = model.wv.most_similar(word1, topn=10)
            neighbors2 = model.wv.most_similar(word2, topn=10)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Соседи **{word1}**:")
                df1 = pd.DataFrame(neighbors1, columns=['Слово', 'Сходство'])
                st.dataframe(df1)
            
            with col2:
                st.write(f"Соседи **{word2}**:")
                df2 = pd.DataFrame(neighbors2, columns=['Слово', 'Сходство'])
                st.dataframe(df2)
                
        except Exception as e:
            st.error(f"Ошибка: {e}")

def semantic_axes_interface(model):
    """Визуализация семантических осей"""
    st.header("📈 Семантические оси")
    
    if model is None:
        st.error("Модель не загружена")
        return
    
    st.info("💡 Используйте слова из вашего корпуса: матбугат, хезмәте, хокук, саклау, эчке, эшләр и т.д.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pos1 = st.text_input("Полюс 1+", "матбугат")
    with col2:
        pos2 = st.text_input("Полюс 2+", "хезмәте")
    with col3:
        neg1 = st.text_input("Полюс 1-", "хокук")
    with col4:
        neg2 = st.text_input("Полюс 2-", "саклау")
    
    test_words = st.text_area("Слова для анализа (через запятую):", 
                             "эчке, эшләр, хәбәр, итә, сум, тәшкил")
    
    if st.button("Построить ось"):
        try:
            # Вычисляем ось
            pos_words = [w for w in [pos1, pos2] if w in model.wv]
            neg_words = [w for w in [neg1, neg2] if w in model.wv]
            
            if not pos_words:
                st.error("Не найдены положительные полюсы")
                return
            if not neg_words:
                st.error("Не найдены отрицательные полюсы")
                return
            
            pos_vecs = [model.wv[w] for w in pos_words]
            neg_vecs = [model.wv[w] for w in neg_words]
            
            axis = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
            axis = axis / np.linalg.norm(axis)
            
            # Проецируем слова
            words_to_test = [w.strip() for w in test_words.split(',')]
            projections = {}
            
            for word in words_to_test:
                if word in model.wv:
                    projections[word] = np.dot(model.wv[word], axis)
            
            if not projections:
                st.error("Ни одно из тестовых слов не найдено в модели")
                return
            
            # Сортируем и визуализируем
            sorted_projections = sorted(projections.items(), key=lambda x: x[1])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            words = [item[0] for item in sorted_projections]
            values = [item[1] for item in sorted_projections]
            colors = ['red' if v < 0 else 'green' for v in values]
            
            ax.barh(words, values, color=colors, alpha=0.6)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Проекция на семантическую ось')
            ax.set_title(f'Ось: {"+".join(pos_words)} - {"+".join(neg_words)}')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            
            # Таблица результатов
            df = pd.DataFrame(sorted_projections, columns=['Слово', 'Проекция'])
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

def visualization_interface(model):
    """2D/3D визуализация"""
    st.header("🎨 2D/3D Визуализация")
    
    if model is None:
        st.error("Модель не загружена")
        return
    
    words_input = st.text_area("Слова для визуализации (через запятую):",
                              "матбугат, хезмәте, хокук, саклау, эчке, эшләр, хәбәр, итә")
    
    method = st.selectbox("Метод проекции:", ["PCA", "t-SNE"])
    dimensions = st.radio("Размерность:", [2, 3])
    
    if st.button("Визуализировать"):
        try:
            words = [w.strip() for w in words_input.split(',')]
            available_words = [w for w in words if w in model.wv]
            
            if len(available_words) < 3:
                st.error(f"Нужно минимум 3 слова. Найдено: {available_words}")
                return
            
            # Получаем векторы
            vectors = np.array([model.wv[word] for word in available_words])
            
            # Проекция
            if method == "PCA":
                projector = PCA(n_components=dimensions)
            else:
                projector = TSNE(n_components=dimensions, random_state=42, perplexity=min(5, len(available_words)-1))
            
            projected = projector.fit_transform(vectors)
            
            # Визуализация
            if dimensions == 2:
                fig, ax = plt.subplots(figsize=(12, 10))
                scatter = ax.scatter(projected[:, 0], projected[:, 1], alpha=0.7, s=100)
                
                for i, word in enumerate(available_words):
                    ax.annotate(word, (projected[i, 0], projected[i, 1]),
                               xytext=(5, 5), textcoords='offset points')
                
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title(f'{method} проекция слов')
                ax.grid(alpha=0.3)
                
            else:  # 3D
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], 
                                   alpha=0.7, s=100)
                
                for i, word in enumerate(available_words):
                    ax.text(projected[i, 0], projected[i, 1], projected[i, 2], word)
                
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                ax.set_title(f'{method} проекция слов')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

def analogy_test_interface(model):
    """Тестирование аналогий"""
    st.header("🧠 Тест аналогий")
    
    if model is None:
        st.error("Модель не загружена")
        return
    
    analogy = st.text_input("Аналогия (формат: слово1 слово2 слово3 ответ):", 
                           "казан татарстан россия мәскәү")
    
    if st.button("Проверить аналогию"):
        try:
            words = analogy.split()
            if len(words) != 4:
                st.error("Нужно 4 слова")
                return
            
            a, b, c, expected = words
            
            # Проверяем наличие слов
            missing = [w for w in [a, b, c, expected] if w not in model.wv]
            if missing:
                st.error(f"Слова не найдены: {missing}")
                return
            
            # Вычисляем аналогию
            result = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
            
            st.subheader("Результат:")
            df = pd.DataFrame(result, columns=['Слово', 'Сходство'])
            st.dataframe(df)
            
            # Проверяем правильность
            predicted_words = [word for word, _ in result]
            is_correct = expected in predicted_words
            
            if is_correct:
                position = predicted_words.index(expected) + 1
                st.success(f"✅ Правильно! Слово '{expected}' на позиции {position}")
            else:
                st.error(f"❌ Ожидалось: '{expected}'")
                
        except Exception as e:
            st.error(f"Ошибка: {e}")

def main():
    """Главная функция"""
    st.set_page_config(page_title="Анализ векторных пространств", layout="wide")
    st.title("🔍 Анализ векторных пространств татарского языка")
    
    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model = load_model()
    
    # Загрузка метаданных
    metadata = load_model_metadata()
    
    # Боковая панель с навигацией
    st.sidebar.title("Навигация")
    section = st.sidebar.radio("Раздел:", [
        "Векторная арифметика",
        "Анализ сходства", 
        "Семантические оси",
        "2D/3D Визуализация",
        "Тест аналогий"
    ])
    
    # Отображение выбранного раздела
    if section == "Векторная арифметика":
        vector_arithmetic_interface(model)
    elif section == "Анализ сходства":
        similarity_analysis_interface(model)
    elif section == "Семантические оси":
        semantic_axes_interface(model)
    elif section == "2D/3D Визуализация":
        visualization_interface(model)
    elif section == "Тест аналогий":
        analogy_test_interface(model)
    
    # Информация о модели
    st.sidebar.markdown("---")
    st.sidebar.subheader("Информация о модели")
    
    if model is not None:
        st.sidebar.write(f"Размер словаря: {len(model.wv.key_to_index):,}")
        st.sidebar.write(f"Размерность: {model.wv.vector_size}")
        st.sidebar.write(f"Тип модели: {type(model).__name__}")
    
    if 'error' not in metadata:
        st.sidebar.write("Метаданные:")
        for key, value in metadata.items():
            if key != 'error':
                st.sidebar.write(f"- {key}: {value}")

if __name__ == "__main__":
    main()