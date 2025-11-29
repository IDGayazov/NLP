from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import umap


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
        Самые частые слова в каждом кластере
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
            if cluster_id == -1:  # пропускаем шум
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

    def visualize_umap(self, labels, title="Визуализация кластеров (UMAP)"):
        """
        Визуализация кластеров в 2D с помощью UMAP
        """

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
        plt.figure(figsize=(10, 8))

        # Разные цвета для каждого кластера
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:
                # Шумовые точки серым цветом
                mask = labels == cluster_id
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c='gray', alpha=0.5, s=20, label=f'Шум ({cluster_id})')
            else:
                mask = labels == cluster_id
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=[colors[i]], alpha=0.7, s=30, label=f'Кластер {cluster_id}')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return embedding_2d

    def visualize_pca(self, labels, title="Визуализация кластеров (PCA)"):
        """
        Визуализация кластеров в 2D с помощью PCA (альтернатива UMAP)
        """
        from sklearn.decomposition import PCA

        if self.X is None:
            self.fit_vectorizer()

        print("🔄 Строим PCA визуализацию...")

        X_dense = self.X.toarray()

        # Создаем PCA редуктор
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(X_dense)

        # Создаем график
        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:
                mask = labels == cluster_id
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c='gray', alpha=0.5, s=20, label=f'Шум ({cluster_id})')
            else:
                mask = labels == cluster_id
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=[colors[i]], alpha=0.7, s=30, label=f'Кластер {cluster_id}')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return embedding_2d


# Обновленная функция сравнения с интерпретацией и визуализацией
def compare_cluster_sizes_with_interpretation(texts, true_labels, max_k=6):
    """
    Сравнение кластеризации с метриками, интерпретацией и визуализацией
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer().toarray()

    print("🔬 СРАВНЕНИЕ РАЗЛИЧНЫХ K (ВНУТРЕННИЕ + ВНЕШНИЕ МЕТРИКИ):")
    print("k\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
    print("-" * 85)

    k_values = []
    internal_metrics = {'silhouette': [], 'calinski': [], 'davies': []}
    external_metrics = {'ari': [], 'nmi': [], 'v_measure': []}

    best_k_internal = 2
    best_k_external = 2
    best_silhouette = -1
    best_ari = -1
    best_labels = None

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # Метрики
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        v_measure = v_measure_score(true_labels, labels)

        print(f"{k}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
              f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")

        # Сохраняем метрики
        k_values.append(k)
        internal_metrics['silhouette'].append(silhouette)
        internal_metrics['calinski'].append(calinski)
        internal_metrics['davies'].append(davies)
        external_metrics['ari'].append(ari)
        external_metrics['nmi'].append(nmi)
        external_metrics['v_measure'].append(v_measure)

        # Находим лучшее k
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k_internal = k
            best_labels = labels

        if ari > best_ari:
            best_ari = ari
            best_k_external = k

    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print(f"   По внутренним метрикам: k={best_k_internal} (Silhouette: {best_silhouette:.3f})")
    print(f"   По внешним метрикам: k={best_k_external} (ARI: {best_ari:.3f})")

    # ИНТЕРПРЕТАЦИЯ ЛУЧШИХ КЛАСТЕРОВ
    if best_labels is not None:
        print(f"\n{'=' * 60}")
        print(f"🔍 ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ (k={best_k_internal})")
        print(f"{'=' * 60}")

        interpreter.print_cluster_info(best_labels)
        interpreter.plot_keywords_barchart(best_labels)

        # ВИЗУАЛИЗАЦИЯ UMAP
        print(f"\n{'=' * 60}")
        print(f"📊 ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print(f"{'=' * 60}")

        # Пробуем UMAP, если не установлен - используем PCA
        umap_embedding = interpreter.visualize_umap(
            best_labels,
            title=f"Кластеризация текстов (k={best_k_internal}) - UMAP"
        )

        if umap_embedding is None:
            # Используем PCA как запасной вариант
            interpreter.visualize_pca(
                best_labels,
                title=f"Кластеризация текстов (k={best_k_internal}) - PCA"
            )

    # Графики метрик
    _plot_all_metrics(k_values, internal_metrics, external_metrics,
                      best_k_internal, best_k_external, texts, true_labels)

    return best_k_internal, best_k_external, best_labels


def _plot_all_metrics(k_values, internal_metrics, external_metrics, best_k_int, best_k_ext, texts, true_labels):
    """
    Построение графиков для всех метрик
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Внутренние метрики
    ax1.plot(k_values, internal_metrics['silhouette'], 'bo-', label='Silhouette', linewidth=2)
    ax1.plot(k_values, _normalize(internal_metrics['calinski']), 'go-', label='Calinski (норм.)', linewidth=2)
    ax1.plot(k_values, _normalize([1 / d for d in internal_metrics['davies']]), 'ro-',
             label='1/Davies (норм.)', linewidth=2)
    ax1.axvline(x=best_k_int, color='blue', linestyle='--', alpha=0.7,
                label=f'Лучшее k={best_k_int} (внутр.)')
    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Нормализованные значения')
    ax1.set_title('ВНУТРЕННИЕ МЕТРИКИ\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Внешние метрики
    ax2.plot(k_values, external_metrics['ari'], 'bo-', label='ARI', linewidth=2)
    ax2.plot(k_values, external_metrics['nmi'], 'go-', label='NMI', linewidth=2)
    ax2.plot(k_values, external_metrics['v_measure'], 'ro-', label='V-measure', linewidth=2)
    ax2.axvline(x=best_k_ext, color='red', linestyle='--', alpha=0.7,
                label=f'Лучшее k={best_k_ext} (внеш.)')
    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Значения метрик')
    ax2.set_title('ВНЕШНИЕ МЕТРИКИ\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Сравнение внутренних и внешних метрик
    ax3.plot(k_values, internal_metrics['silhouette'], 'b-', label='Silhouette', linewidth=2)
    ax3.plot(k_values, external_metrics['ari'], 'r-', label='ARI', linewidth=2)
    ax3.axvline(x=best_k_int, color='blue', linestyle='--', alpha=0.5,
                label=f'Лучшее k (внутр.)={best_k_int}')
    ax3.axvline(x=best_k_ext, color='red', linestyle='--', alpha=0.5,
                label=f'Лучшее k (внеш.)={best_k_ext}')
    ax3.set_xlabel('Количество кластеров (k)')
    ax3.set_ylabel('Значения метрик')
    ax3.set_title('СРАВНЕНИЕ: Silhouette vs ARI')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Homogeneity vs Completeness
    homogeneity_scores = []
    completeness_scores = []

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        homogeneity_scores.append(homogeneity_score(true_labels, labels))
        completeness_scores.append(completeness_score(true_labels, labels))

    ax4.plot(k_values, homogeneity_scores, 'b-', label='Homogeneity', linewidth=2)
    ax4.plot(k_values, completeness_scores, 'g-', label='Completeness', linewidth=2)
    ax4.plot(k_values, external_metrics['v_measure'], 'r-', label='V-measure', linewidth=2)
    ax4.set_xlabel('Количество кластеров (k)')
    ax4.set_ylabel('Значения метрик')
    ax4.set_title('HOMOGENEITY, COMPLETENESS, V-MEASURE')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def _normalize(values):
    """Нормализация значений к диапазону [0, 1]"""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


# Простая функция для быстрой интерпретации с визуализацией
def quick_interpret_clusters(texts, labels, n_words=8):
    """
    Быстрая интерпретация кластеров с визуализацией
    """
    interpreter = SimpleClusterInterpreter(texts)
    interpreter.fit_vectorizer()

    print("📊 ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ")
    print("=" * 50)
    interpreter.print_cluster_info(labels, n_words)
    interpreter.plot_keywords_barchart(labels, n_words)

    # Визуализация
    print("\n📈 ВИЗУАЛИЗАЦИЯ")
    print("=" * 50)
    interpreter.visualize_umap(labels)
    interpreter.visualize_pca(labels)


if __name__ == "__main__":
    from util.decribe import get_texts, get_labels

    texts = get_texts()
    true_labels = get_labels()

    print("🚀 СРАВНЕНИЕ КЛАСТЕРОВ С ИНТЕРПРЕТАЦИЕЙ И ВИЗУАЛИЗАЦИЕЙ")
    print("=" * 70)

    best_k_int, best_k_ext, best_labels = compare_cluster_sizes_with_interpretation(
        texts, true_labels, max_k=5
    )