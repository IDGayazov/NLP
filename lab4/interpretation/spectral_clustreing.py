from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP не установлен. Установите: pip install umap-learn")

from util.decribe import get_labels, get_texts


class SimpleClusterInterpreter:
    """
    Простой интерпретатор кластеров для TF-IDF
    """

    def __init__(self, texts, vectorizer=None):
        self.texts = texts
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=500)
        self.X = None
        self.feature_names = None

    def fit_vectorizer(self):
        """Обучение векторизатора"""
        self.X = self.vectorizer.fit_transform(self.texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.X

    def get_cluster_keywords(self, labels, n_words=10):
        """
        Топ-N ключевых слов для каждого кластера по TF-IDF
        """
        if self.X is None:
            self.fit_vectorizer()

        unique_labels = np.unique(labels)
        cluster_keywords = {}

        for cluster_id in unique_labels:
            # Индексы документов в кластере
            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                cluster_keywords[cluster_id] = []
                continue

            # Средние TF-IDF веса для кластера
            cluster_tfidf = self.X[cluster_indices].mean(axis=0)
            cluster_tfidf = np.array(cluster_tfidf).flatten()

            # Топ-N слов с наибольшими весами
            top_indices = np.argsort(cluster_tfidf)[::-1][:n_words]
            top_words = [(self.feature_names[i], cluster_tfidf[i])
                         for i in top_indices if cluster_tfidf[i] > 0]

            cluster_keywords[cluster_id] = top_words

        return cluster_keywords

    def get_most_frequent_words(self, labels, n_words=10):
        """
        Самые частые слова в каждом кластерах
        """
        unique_labels = np.unique(labels)
        cluster_freq_words = {}

        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [self.texts[i] for i in cluster_indices]

            # Собираем все слова кластера
            all_words = []
            for text in cluster_texts:
                words = text.lower().split()
                words = [word for word in words if len(word) > 2]  # убираем короткие слова
                all_words.extend(words)

            # Самые частые слова
            word_counts = Counter(all_words)
            cluster_freq_words[cluster_id] = word_counts.most_common(n_words)

        return cluster_freq_words

    def print_cluster_info(self, labels, n_words=8):
        """
        Красивая печать информации о кластерах
        """
        unique_labels, counts = np.unique(labels, return_counts=True)

        print("📊 ИНФОРМАЦИЯ О КЛАСТЕРАХ:")
        print("=" * 60)

        # Топ слова по TF-IDF
        tfidf_keywords = self.get_cluster_keywords(labels, n_words)

        # Частотные слова
        freq_words = self.get_most_frequent_words(labels, n_words)

        for cluster_id in unique_labels:
            count = counts[unique_labels == cluster_id][0]
            percentage = (count / len(labels)) * 100

            print(f"\n🔸 КЛАСТЕР {cluster_id} ({count} документов, {percentage:.1f}%):")

            # TF-IDF ключевые слова
            if cluster_id in tfidf_keywords:
                tfidf_words = [word for word, weight in tfidf_keywords[cluster_id][:5]]
                print(f"   📈 Ключевые слова (TF-IDF): {', '.join(tfidf_words)}")

            # Частотные слова
            if cluster_id in freq_words:
                freq_words_list = [word for word, count in freq_words[cluster_id][:5]]
                print(f"   📊 Частые слова: {', '.join(freq_words_list)}")

            # Примеры документов
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                sample_text = self.texts[cluster_indices[0]]
                preview = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
                print(f"   📄 Пример: {preview}")

    def plot_keywords_barchart(self, labels, n_words=6):
        """
        Визуализация ключевых слов кластеров
        """
        cluster_keywords = self.get_cluster_keywords(labels, n_words)
        n_clusters = len(cluster_keywords)

        if n_clusters == 0:
            print("❌ Нет кластеров для визуализации")
            return

        fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5))
        if n_clusters == 1:
            axes = [axes]

        for idx, (cluster_id, words_weights) in enumerate(cluster_keywords.items()):
            if not words_weights:
                continue

            words, weights = zip(*words_weights)

            axes[idx].barh(range(len(words)), weights, color=f'C{idx}', alpha=0.7)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words, fontsize=9)
            axes[idx].set_title(f'Кластер {cluster_id}')
            axes[idx].set_xlabel('TF-IDF вес')

        plt.suptitle('КЛЮЧЕВЫЕ СЛОВА КЛАСТЕРОВ', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_umap(self, labels, title="Spectral Clustering - Визуализация кластеров (UMAP)"):
        """
        Визуализация кластеров в 2D с помощью UMAP
        """
        if not UMAP_AVAILABLE:
            print("❌ UMAP не установлен. Используйте: pip install umap-learn")
            return None

        if self.X is None:
            self.fit_vectorizer()

        print("🔄 Строим UMAP визуализацию...")

        # Преобразуем в плотный массив для UMAP
        X_dense = self.X.toarray()

        # Создаем UMAP редуктор
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )

        # Уменьшаем размерность
        embedding_2d = reducer.fit_transform(X_dense)

        # Создаем график
        plt.figure(figsize=(12, 8))

        # Разные цвета для каждого кластера
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                        c=[colors[i]], alpha=0.7, s=30, label=f'Кластер {cluster_id} ({np.sum(mask)} точек)')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return embedding_2d


def simple_spectral_cluster(texts, n_clusters=3, affinity='cosine', n_neighbors=10):
    """
    Упрощенная кластеризация SpectralClustering с интерпретацией
    """
    # Инициализация интерпретатора
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()

    print(f"🔄 Запуск SpectralClustering с {n_clusters} кластерами...")
    print(f"⚙️  Параметры: affinity={affinity}, n_neighbors={n_neighbors}")

    # SpectralClustering кластеризация
    if affinity == 'nearest_neighbors':
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            random_state=42,
            n_init=5
        )
    else:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=42,
            n_init=5
        )

    # Выбираем правильный формат данных
    if affinity == 'cosine':
        X_used = X  # Для cosine используем разреженную матрицу
    else:
        X_used = X_dense  # Для других affinity используем плотную матрицу

    labels = spectral.fit_predict(X_used)

    # Вычисляем метрики
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(X_dense, labels)

    # Вывод результатов
    print(f"\n📊 SPECTRAL CLUSTERING")
    print("=" * 50)
    print(f"⚙️  ПАРАМЕТРЫ: n_clusters={n_clusters}, affinity={affinity}")
    if affinity == 'nearest_neighbors':
        print(f"                 n_neighbors={n_neighbors}")
    print(f"🎯 РЕЗУЛЬТАТЫ:")
    print(f"   • Всего документов: {len(texts)}")
    print(f"   • Silhouette Score: {silhouette:.3f}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    for cluster_id in unique_labels:
        count = counts[unique_labels == cluster_id][0]
        percentage = (count / len(texts)) * 100
        print(f"   • Кластер {cluster_id}: {count} документов ({percentage:.1f}%)")

    # Информация о типах сходства
    print(f"\n📋 ТИПЫ СХОДСТВА (AFFINITY):")
    affinity_info = {
        'rbf': "Радиальная базисная функция (гауссово ядро)",
        'nearest_neighbors': "Граф ближайших соседей",
        'cosine': "Косинусное сходство (лучше для текстов)"
    }
    print(f"   {affinity}: {affinity_info.get(affinity, '')}")

    # Особенности Spectral Clustering
    print(f"\n💡 ОСОБЕННОСТИ SPECTRAL CLUSTERING:")
    print("   • Работает с данными сложной невыпуклой формы")
    print("   • Использует спектральную теорию графов")
    print("   • Эффективен когда кластеры не разделимы линейно")
    print("   • Чувствителен к выбору функции сходства")

    # Интерпретация кластеров
    print(f"\n{'=' * 50}")
    interpreter.print_cluster_info(labels)

    # Визуализация ключевых слов
    print(f"\n{'=' * 50}")
    print("📈 ВИЗУАЛИЗАЦИЯ КЛЮЧЕВЫХ СЛОВ")
    interpreter.plot_keywords_barchart(labels)

    # UMAP визуализация
    print(f"\n{'=' * 50}")
    print("🎨 ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
    interpreter.visualize_umap(labels, f"Spectral Clustering (k={n_clusters}, affinity={affinity})")

    return labels


def quick_spectral_analysis(texts, n_clusters_values=[2, 3, 4], affinity_types=['cosine', 'rbf']):
    """
    Быстрый анализ SpectralClustering с разными параметрами
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()

    print("🚀 БЫСТРЫЙ АНАЛИЗ SPECTRAL CLUSTERING")
    print("=" * 70)
    print("k\tAffinity\tSilhouette\tВремя (сек)")
    print("-" * 60)

    results = []

    for n_clusters in n_clusters_values:
        for affinity in affinity_types:
            try:
                # Замеряем время
                import time
                start_time = time.time()

                # Настраиваем параметры
                if affinity == 'nearest_neighbors':
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        n_neighbors=10,
                        random_state=42,
                        n_init=5
                    )
                else:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        random_state=42,
                        n_init=5
                    )

                # Выбираем формат данных
                if affinity == 'cosine':
                    X_used = X
                else:
                    X_used = X_dense

                labels = spectral.fit_predict(X_used)

                execution_time = time.time() - start_time

                # Вычисляем silhouette score
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X_dense, labels)

                print(f"{n_clusters}\t{affinity}\t\t{silhouette:.3f}\t\t{execution_time:.2f}")

                results.append({
                    'n_clusters': n_clusters,
                    'affinity': affinity,
                    'silhouette': silhouette,
                    'time': execution_time,
                    'labels': labels
                })

            except Exception as e:
                print(f"{n_clusters}\t{affinity}\t\tERROR\t\t-")

    # Находим лучшую конфигурацию
    if results:
        best_result = max(results, key=lambda x: x['silhouette'])
        print(f"\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
        print(f"   n_clusters={best_result['n_clusters']}, affinity={best_result['affinity']}")
        print(f"   Silhouette Score: {best_result['silhouette']:.3f}")
        print(f"   Время выполнения: {best_result['time']:.2f} сек")

        # Интерпретация лучшего результата
        print(f"\n{'=' * 50}")
        print("🔍 ИНТЕРПРЕТАЦИЯ ЛУЧШИХ КЛАСТЕРОВ")
        interpreter.print_cluster_info(best_result['labels'])

        return best_result['labels']

    return None

def compare_affinity_types(texts, n_clusters=3):
    """
    Сравнение разных типов сходства для SpectralClustering
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()

    affinity_types = ['cosine', 'rbf', 'nearest_neighbors']
    affinity_info = {
        'cosine': "Косинусное сходство (лучше для текстов)",
        'rbf': "Гауссово ядро (евклидово расстояние)",
        'nearest_neighbors': "Граф ближайших соседей"
    }

    print("🔬 СРАВНЕНИЕ ТИПОВ СХОДСТВА (AFFINITY)")
    print("=" * 70)
    print("Affinity\t\tОписание\t\t\tSilhouette")
    print("-" * 80)

    results = {}

    for affinity in affinity_types:
        try:
            # Настраиваем параметры
            if affinity == 'nearest_neighbors':
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    n_neighbors=10,
                    random_state=42,
                    n_init=5
                )
            else:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    random_state=42,
                    n_init=5
                )

            # Выбираем формат данных
            if affinity == 'cosine':
                X_used = X
            else:
                X_used = X_dense

            labels = spectral.fit_predict(X_used)

            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_dense, labels)

            description = affinity_info.get(affinity, "")
            print(f"{affinity}\t\t{description[:25]}\t{silhouette:.3f}")

            results[affinity] = {
                'labels': labels,
                'silhouette': silhouette,
                'description': description
            }

        except Exception as e:
            print(f"{affinity}\t\t{affinity_info.get(affinity, '')[:25]}\tERROR")

    return results


def analyze_similarity_structure(texts):
    """
    Анализ структуры сходства данных
    """
    from sklearn.metrics.pairwise import cosine_similarity

    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()

    print("🔗 АНАЛИЗ СТРУКТУРЫ СХОДСТВА ДАННЫХ")
    print("=" * 50)

    # Косинусное сходство
    similarities = cosine_similarity(X)
    np.fill_diagonal(similarities, 0)  # Игнорируем самосходство

    print(f"📊 СТАТИСТИКА КОСИНУСНОГО СХОДСТВА:")
    print(f"   • Среднее сходство: {np.mean(similarities):.4f}")
    print(f"   • Максимальное сходство: {np.max(similarities):.4f}")
    print(f"   • Минимальное сходство: {np.min(similarities):.4f}")
    print(f"   • Стандартное отклонение: {np.std(similarities):.4f}")

    # Анализ распределения сходства
    high_similarity = np.sum(similarities > 0.7) / (len(similarities) ** 2) * 100
    medium_similarity = np.sum((similarities > 0.3) & (similarities <= 0.7)) / (len(similarities) ** 2) * 100
    low_similarity = np.sum(similarities <= 0.3) / (len(similarities) ** 2) * 100

    print(f"\n📈 РАСПРЕДЕЛЕНИЕ СХОДСТВА:")
    print(f"   • Высокое сходство (>0.7): {high_similarity:.1f}% пар")
    print(f"   • Среднее сходство (0.3-0.7): {medium_similarity:.1f}% пар")
    print(f"   • Низкое сходство (≤0.3): {low_similarity:.1f}% пар")


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 SPECTRAL CLUSTERING ДЛЯ ТЕКСТОВ С ИНТЕРПРЕТАЦИЕЙ")
    print("=" * 70)

    # Анализ структуры данных
    print("🎯 АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
    analyze_similarity_structure(texts)

    # Вариант 1: Простая кластеризация с интерпретацией
    print(f"\n{'=' * 70}")
    print("🎯 ВАРИАНТ 1: ПРОСТАЯ КЛАСТЕРИЗАЦИЯ С ИНТЕРПРЕТАЦИЕЙ")
    labels1 = simple_spectral_cluster(texts, n_clusters=3, affinity='cosine')

    # Вариант 2: Сравнение типов сходства
    print(f"\n{'=' * 70}")
    print("🎯 ВАРИАНТ 2: СРАВНЕНИЕ ТИПОВ СХОДСТВА")
    affinity_results = compare_affinity_types(texts, n_clusters=3)

    # Визуализация лучшего типа сходства
    if affinity_results:
        best_affinity = max(affinity_results.keys(),
                            key=lambda x: affinity_results[x]['silhouette']
                            if 'silhouette' in affinity_results[x] else -1)

        print(f"\n🎨 ВИЗУАЛИЗАЦИЯ ДЛЯ ЛУЧШЕГО AFFINITY: {best_affinity}")
        interpreter = SimpleClusterInterpreter(texts)
        interpreter.fit_vectorizer()
        interpreter.visualize_umap(
            affinity_results[best_affinity]['labels'],
            f"Spectral Clustering (affinity={best_affinity})"
        )

    # Вариант 3: Быстрый анализ с разными k
    print(f"\n{'=' * 70}")
    print("🎯 ВАРИАНТ 3: БЫСТРЫЙ АНАЛИЗ С РАЗНЫМИ K")
    best_labels = quick_spectral_analysis(
        texts,
        n_clusters_values=[2, 3, 4],
        affinity_types=['cosine', 'rbf']
    )