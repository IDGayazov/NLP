from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

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
            # Пропускаем шумовые точки
            if cluster_id == -1:
                continue

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
        Самые частые слова в каждом кластере
        """
        unique_labels = np.unique(labels)
        cluster_freq_words = {}

        for cluster_id in unique_labels:
            # Пропускаем шумовые точки
            if cluster_id == -1:
                continue

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

        # Сначала выводим шум
        noise_count = counts[unique_labels == -1][0] if -1 in unique_labels else 0
        if noise_count > 0:
            print(f"\n🔸 ШУМ: {noise_count} документов ({noise_count / len(labels) * 100:.1f}%)")

        # Затем кластеры
        for cluster_id in unique_labels:
            if cluster_id == -1:  # шум уже вывели
                continue

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

    def visualize_umap(self, labels, title="DBSCAN - Визуализация кластеров (UMAP)"):
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
            if cluster_id == -1:
                # Шумовые точки серым цветом
                mask = labels == cluster_id
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c='gray', alpha=0.5, s=20, label=f'Шум ({np.sum(mask)} точек)')
            else:
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


def find_optimal_eps(texts, k=5):
    """
    Поиск оптимального eps с помощью k-расстояний
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    # Используем косинусное расстояние
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    distances = np.sort(distances[:, k - 1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Точки (отсортированные по расстоянию)')
    plt.ylabel(f'{k}-е расстояние')
    plt.title('МЕТОД КОЛЕНА ДЛЯ ОПРЕДЕЛЕНИЯ EPS\n(ищем точку изгиба)')
    plt.grid(True, alpha=0.3)

    # Автоматическое определение точки изгиба
    gradients = np.gradient(distances)
    elbow_point = np.argmax(gradients) + 1

    plt.axvline(x=elbow_point, color='red', linestyle='--',
                label=f'Точка изгиба (eps ≈ {distances[elbow_point]:.3f})')
    plt.legend()
    plt.show()

    recommended_eps = distances[elbow_point]
    print(f"🎯 РЕКОМЕНДУЕМЫЙ EPS: {recommended_eps:.3f}")

    return recommended_eps


def simple_dbscan_cluster(texts, eps=0.3, min_samples=3):
    """
    Упрощенная кластеризация DBSCAN с интерпретацией
    """
    # Инициализация интерпретатора
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()

    # DBSCAN кластеризация
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine'
    )
    labels = dbscan.fit_predict(X)

    # Анализ результатов
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    # Вывод результатов
    print(f"📊 DBSCAN КЛАСТЕРИЗАЦИЯ")
    print("=" * 50)
    print(f"⚙️  ПАРАМЕТРЫ: eps={eps}, min_samples={min_samples}")
    print(f"🎯 РЕЗУЛЬТАТЫ:")
    print(f"   • Количество кластеров: {n_clusters}")
    print(f"   • Шумовых точек: {n_noise} ({n_noise / len(texts) * 100:.1f}%)")
    print(f"   • Всего документов: {len(texts)}")

    if n_clusters == 0:
        print("❌ Не найдено кластеров. Попробуйте уменьшить eps или min_samples")
        return labels

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
    interpreter.visualize_umap(labels, f"DBSCAN кластеризация (eps={eps}, min_samples={min_samples})")

    return labels


def quick_dbscan_analysis(texts, eps_values=[0.2, 0.3, 0.4], min_samples_values=[2, 3]):
    """
    Быстрый анализ DBSCAN с разными параметрами
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()

    print("🚀 БЫСТРЫЙ АНАЛИЗ DBSCAN С РАЗНЫМИ ПАРАМЕТРАМИ")
    print("=" * 60)
    print("eps\tmin_sam\tClusters\tNoise\t% шума")
    print("-" * 50)

    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(X)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            noise_percentage = n_noise / len(texts) * 100

            print(f"{eps}\t{min_samples}\t{n_clusters}\t\t{n_noise}\t{noise_percentage:.1f}%")

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })

    # Находим лучшую конфигурацию (минимум шума, хотя бы 2 кластера)
    valid_results = [r for r in results if r['n_clusters'] >= 2]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['n_noise'])
        print(f"\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
        print(f"   eps={best_result['eps']}, min_samples={best_result['min_samples']}")
        print(
            f"   Кластеров: {best_result['n_clusters']}, Шум: {best_result['n_noise']} ({best_result['n_noise'] / len(texts) * 100:.1f}%)")

        # Интерпретация лучшего результата
        print(f"\n{'=' * 50}")
        print("🔍 ИНТЕРПРЕТАЦИЯ ЛУЧШИХ КЛАСТЕРОВ")
        interpreter.print_cluster_info(best_result['labels'])

        return best_result['labels']

    return None


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 DBSCAN ДЛЯ ТЕКСТОВ С ИНТЕРПРЕТАЦИЕЙ")
    print("=" * 60)

    # Вариант 1: Автоматический подбор eps
    print("🎯 ВАРИАНТ 1: АВТОМАТИЧЕСКИЙ ПОДБОР EPS")
    recommended_eps = find_optimal_eps(texts, k=5)

    # Кластеризация с рекомендованным eps
    print(f"\n🎯 КЛАСТЕРИЗАЦИЯ С EPS={recommended_eps:.3f}")
    labels1 = simple_dbscan_cluster(texts, eps=recommended_eps, min_samples=3)

    # Вариант 2: Быстрый анализ с разными параметрами
    print(f"\n{'=' * 60}")
    print("🎯 ВАРИАНТ 2: БЫСТРЫЙ АНАЛИЗ С РАЗНЫМИ ПАРАМЕТРАМИ")
    labels2 = quick_dbscan_analysis(texts, eps_values=[0.2, 0.3, 0.4, 0.5], min_samples_values=[2, 3])