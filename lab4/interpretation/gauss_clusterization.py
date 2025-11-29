from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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

    def visualize_umap(self, labels, title="Gaussian Mixture - Визуализация кластеров (UMAP)"):
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


def simple_gmm_cluster(texts, n_components=3, covariance_type='full'):
    """
    Упрощенная кластеризация GaussianMixture с интерпретацией
    """
    # Инициализация интерпретатора
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()

    # Стандартизация для GaussianMixture
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    print(f"🔄 Запуск GaussianMixture с {n_components} компонентами...")

    # GaussianMixture кластеризация
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        max_iter=100
    )

    # Мягкое назначение - вероятности принадлежности
    soft_labels = gmm.fit_predict(X_scaled)
    # Жесткое назначение для метрик
    hard_labels = gmm.predict(X_scaled)
    # Вероятности принадлежности к кластерам
    probabilities = gmm.predict_proba(X_scaled)

    # Проверяем сходимость
    converged = gmm.converged_
    n_iter = gmm.n_iter_

    # Вывод результатов
    print(f"\n📊 GAUSSIAN MIXTURE (МЯГКАЯ КЛАСТЕРИЗАЦИЯ)")
    print("=" * 60)
    print(f"⚙️  ПАРАМЕТРЫ: n_components={n_components}, covariance_type={covariance_type}")
    print(f"🎯 РЕЗУЛЬТАТЫ:")
    print(f"   • Всего документов: {len(texts)}")
    print(f"   • Сходимость: {'✅ Успешно' if converged else '❌ Не сошлась'}")
    print(f"   • Итераций: {n_iter}")

    unique_labels, counts = np.unique(hard_labels, return_counts=True)
    for cluster_id in unique_labels:
        count = counts[unique_labels == cluster_id][0]
        percentage = (count / len(texts)) * 100
        print(f"   • Кластер {cluster_id}: {count} документов ({percentage:.1f}%)")

    # Анализ уверенности классификации
    max_probs = np.max(probabilities, axis=1)
    confidence_stats = {
        'high_confidence': np.sum(max_probs > 0.9) / len(max_probs) * 100,
        'medium_confidence': np.sum((max_probs > 0.7) & (max_probs <= 0.9)) / len(max_probs) * 100,
        'low_confidence': np.sum(max_probs <= 0.7) / len(max_probs) * 100
    }

    print(f"\n🎯 УВЕРЕННОСТЬ КЛАССИФИКАЦИИ:")
    print(f"   • Высокая уверенность (>0.9): {confidence_stats['high_confidence']:.1f}% точек")
    print(f"   • Средняя уверенность (0.7-0.9): {confidence_stats['medium_confidence']:.1f}% точек")
    print(f"   • Низкая уверенность (≤0.7): {confidence_stats['low_confidence']:.1f}% точек")

    # Информация о типах ковариационных матриц
    print(f"\n📋 ТИПЫ КОВАРИАЦИОННЫХ МАТРИЦ:")
    cov_info = {
        'full': "Полная ковариационная матрица для каждого кластера",
        'tied': "Одна общая ковариационная матрица для всех кластеров",
        'diag': "Диагональная ковариационная матрица для каждого кластера",
        'spherical': "Сферическая ковариационная матрица (одинаковая по всем направлениям)"
    }
    print(f"   {covariance_type}: {cov_info.get(covariance_type, '')}")

    # Интерпретация кластеров (по жесткому назначению)
    print(f"\n{'=' * 50}")
    interpreter.print_cluster_info(hard_labels)

    # Визуализация ключевых слов
    print(f"\n{'=' * 50}")
    print("📈 ВИЗУАЛИЗАЦИЯ КЛЮЧЕВЫХ СЛОВ")
    interpreter.plot_keywords_barchart(hard_labels)

    # UMAP визуализация
    print(f"\n{'=' * 50}")
    print("🎨 ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
    interpreter.visualize_umap(hard_labels,
                               f"Gaussian Mixture (k={n_components}, covariance={covariance_type})")

    # Примеры мягкого назначения
    print(f"\n{'=' * 50}")
    print("🔮 ПРИМЕРЫ МЯГКОГО НАЗНАЧЕНИЯ")
    print("Первые 3 документа и их вероятности принадлежности к кластерам:")
    print("Док.\t" + "\t".join([f"Кл.{i}" for i in range(n_components)]))
    for i in range(min(3, len(probabilities))):
        prob_str = "\t".join([f"{p:.3f}" for p in probabilities[i]])
        assigned_cluster = hard_labels[i]
        print(f"{i}\t{prob_str} → Кластер {assigned_cluster}")

    return hard_labels, soft_labels, probabilities, gmm


def quick_gmm_analysis(texts, n_components_values=[2, 3, 4], covariance_types=['full', 'tied', 'diag']):
    """
    Быстрый анализ GaussianMixture с разными параметрами
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    print("🚀 БЫСТРЫЙ АНАЛИЗ GAUSSIAN MIXTURE")
    print("=" * 70)
    print("k\tCovariance\tConverged\tSilhouette\tBIC")
    print("-" * 70)

    results = []

    for n_components in n_components_values:
        for covariance_type in covariance_types:
            try:
                # Замеряем время
                import time
                start_time = time.time()

                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=42,
                    max_iter=50
                )
                hard_labels = gmm.fit_predict(X_scaled)

                execution_time = time.time() - start_time

                # Проверяем сходимость
                converged = gmm.converged_

                # Вычисляем метрики
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X_dense, hard_labels)
                bic = gmm.bic(X_scaled)

                status = "✅" if converged else "❌"
                print(f"{n_components}\t{covariance_type}\t\t{status}\t\t{silhouette:.3f}\t\t{bic:.0f}")

                results.append({
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'silhouette': silhouette,
                    'bic': bic,
                    'converged': converged,
                    'time': execution_time,
                    'hard_labels': hard_labels,
                    'gmm': gmm
                })

            except Exception as e:
                print(f"{n_components}\t{covariance_type}\t\tERROR\t\t-\t\t-")

    # Находим лучшую конфигурацию (только сходившиеся модели)
    converged_results = [r for r in results if r['converged']]
    if converged_results:
        best_by_silhouette = max(converged_results, key=lambda x: x['silhouette'])
        best_by_bic = min(converged_results, key=lambda x: x['bic'])

        print(f"\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
        print(f"   По Silhouette: k={best_by_silhouette['n_components']}, "
              f"covariance={best_by_silhouette['covariance_type']} "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f})")
        print(f"   По BIC: k={best_by_bic['n_components']}, "
              f"covariance={best_by_bic['covariance_type']} "
              f"(BIC: {best_by_bic['bic']:.0f})")

        # Используем лучшую по Silhouette для интерпретации
        print(f"\n{'=' * 50}")
        print("🔍 ИНТЕРПРЕТАЦИЯ ЛУЧШИХ КЛАСТЕРОВ")
        interpreter.print_cluster_info(best_by_silhouette['hard_labels'])

        return best_by_silhouette['hard_labels'], best_by_silhouette['gmm']

    return None, None


def compare_covariance_types(texts, n_components=3):
    """
    Сравнение разных типов ковариационных матриц
    """
    interpreter = SimpleClusterInterpreter(texts)
    X = interpreter.fit_vectorizer()
    X_dense = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    covariance_types = ['full', 'tied', 'diag', 'spherical']
    cov_info = {
        'full': "Полная матрица для каждого кластера",
        'tied': "Общая матрица для всех кластеров",
        'diag': "Диагональная матрица",
        'spherical': "Сферическая матрица"
    }

    print("🔬 СРАВНЕНИЕ ТИПОВ КОВАРИАЦИОННЫХ МАТРИЦ")
    print("=" * 70)
    print("Covariance\tОписание\t\t\tConverged\tSilhouette")
    print("-" * 80)

    results = {}

    for cov_type in covariance_types:
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=42,
                max_iter=50
            )
            hard_labels = gmm.fit_predict(X_scaled)

            converged = gmm.converged_
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_dense, hard_labels)

            description = cov_info.get(cov_type, "")
            status = "✅" if converged else "❌"
            print(f"{cov_type}\t\t{description[:25]}\t{status}\t\t{silhouette:.3f}")

            results[cov_type] = {
                'hard_labels': hard_labels,
                'silhouette': silhouette,
                'converged': converged,
                'gmm': gmm
            }

        except Exception as e:
            print(f"{cov_type}\t\t{cov_info.get(cov_type, '')[:25]}\tERROR\t\t-")

    return results


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 GAUSSIAN MIXTURE ДЛЯ ТЕКСТОВ С ИНТЕРПРЕТАЦИЕЙ")
    print("=" * 70)

    # Вариант 1: Простая кластеризация с интерпретацией
    print("🎯 ВАРИАНТ 1: ПРОСТАЯ КЛАСТЕРИЗАЦИЯ С ИНТЕРПРЕТАЦИЕЙ")
    hard_labels1, soft_labels1, probabilities1, gmm1 = simple_gmm_cluster(
        texts, n_components=3, covariance_type='full'
    )