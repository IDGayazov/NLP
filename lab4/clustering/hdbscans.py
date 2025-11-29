import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score)

from util.decribe import get_labels, get_texts


def compare_hdbscan_parameters(texts, true_labels=None, min_cluster_size_range=None, min_samples_range=None):
    """
    Сравнение HDBSCAN с различными параметрами min_cluster_size и min_samples
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    has_true_labels = true_labels is not None

    if min_cluster_size_range is None:
        min_cluster_size_range = [2, 3, 5, 10]
    if min_samples_range is None:
        min_samples_range = [1, 2, 3]

    if has_true_labels:
        print("🔬 HDBSCAN С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("min_clust\tmin_sam\tClusters\tNoise\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
        print("-" * 105)
    else:
        print("🔬 HDBSCAN С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("min_clust\tmin_sam\tClusters\tNoise\tSilhouette\tCalinski\tDavies-B")
        print("-" * 75)

    results = []

    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            # HDBSCAN кластеризация
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='cosine',  # Используем косинусное расстояние для текстов
                cluster_selection_epsilon=0.0
            )
            labels = clusterer.fit_predict(X)

            # Игнорируем шумовые точки (-1) для внутренних метрик
            non_noise_mask = labels != -1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            # Если есть как минимум 2 кластера и не все точки - шум
            if n_clusters >= 2 and np.sum(non_noise_mask) >= 2:
                # Внутренние метрики (без шума)
                silhouette = silhouette_score(X_dense[non_noise_mask], labels[non_noise_mask])
                calinski = calinski_harabasz_score(X_dense[non_noise_mask], labels[non_noise_mask])
                davies = davies_bouldin_score(X_dense[non_noise_mask], labels[non_noise_mask])
            else:
                silhouette = calinski = davies = -1

            if has_true_labels:
                # Внешние метрики (включая шум как отдельный кластер)
                ari = adjusted_rand_score(true_labels, labels)
                nmi = normalized_mutual_info_score(true_labels, labels)
                v_measure = v_measure_score(true_labels, labels)

                print(f"{min_cluster_size}\t\t{min_samples}\t{n_clusters}\t\t{n_noise}\t{silhouette:.3f}\t\t"
                      f"{calinski:.3f}\t\t{davies:.3f}\t\t{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
            else:
                print(f"{min_cluster_size}\t\t{min_samples}\t{n_clusters}\t\t{n_noise}\t{silhouette:.3f}\t\t"
                      f"{calinski:.3f}\t\t{davies:.3f}")

            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'ari': ari if has_true_labels else -1,
                'nmi': nmi if has_true_labels else -1,
                'v_measure': v_measure if has_true_labels else -1,
                'labels': labels,
                'clusterer': clusterer
            })

    # Находим лучшие параметры
    valid_results = [r for r in results if r['n_clusters'] >= 2]

    if valid_results:
        best_by_silhouette = max(valid_results, key=lambda x: x['silhouette'])
        if has_true_labels:
            best_by_ari = max(valid_results, key=lambda x: x['ari'])

        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print(f"   По внутренним метрикам: min_cluster_size={best_by_silhouette['min_cluster_size']}, "
              f"min_samples={best_by_silhouette['min_samples']} "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f}, "
              f"Кластеров: {best_by_silhouette['n_clusters']})")
        if has_true_labels:
            print(f"   По внешним метрикам: min_cluster_size={best_by_ari['min_cluster_size']}, "
                  f"min_samples={best_by_ari['min_samples']} "
                  f"(ARI: {best_by_ari['ari']:.3f}, "
                  f"Кластеров: {best_by_ari['n_clusters']})")
    else:
        print(f"\n⚠️  Не найдено параметров, создающих хотя бы 2 кластера")
        best_by_silhouette = best_by_ari = None

    # Строим графики
    if has_true_labels and valid_results:
        _plot_hdbscan_metrics(results, texts, true_labels, best_by_silhouette, best_by_ari)
    elif valid_results:
        _plot_hdbscan_internal_metrics(results, best_by_silhouette)

    if has_true_labels and valid_results:
        return best_by_silhouette, best_by_ari
    elif valid_results:
        return best_by_silhouette
    else:
        return None, None if has_true_labels else None


def _plot_hdbscan_metrics(results, texts, true_labels, best_silhouette, best_ari):
    """
    Построение графиков для HDBSCAN со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Подготовка данных для графиков
    min_cluster_sizes = sorted(set(r['min_cluster_size'] for r in results))
    min_samples_values = sorted(set(r['min_samples'] for r in results))

    # График 1: Количество кластеров и шума для разных min_cluster_size
    cluster_data_by_size = {}
    for min_cluster_size in min_cluster_sizes:
        size_data = [r for r in results if r['min_cluster_size'] == min_cluster_size and r['min_samples'] == 1]
        if size_data:
            cluster_data_by_size[min_cluster_size] = size_data[0]

    if cluster_data_by_size:
        sizes = list(cluster_data_by_size.keys())
        clusters = [cluster_data_by_size[size]['n_clusters'] for size in sizes]
        noise = [cluster_data_by_size[size]['n_noise'] for size in sizes]

        ax1.plot(sizes, clusters, 'bo-', linewidth=2, markersize=6, label='Кластеры')
        ax1.plot(sizes, noise, 'ro-', linewidth=2, markersize=6, label='Шум')

    ax1.set_xlabel('min_cluster_size')
    ax1.set_ylabel('Количество')
    ax1.set_title('HDBSCAN: КЛАСТЕРЫ И ШУМ\n(min_samples=1)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Внутренние метрики для разных min_samples
    for min_cluster_size in [2, 5]:  # Показываем для двух значений min_cluster_size
        samples_data = [r for r in results if r['min_cluster_size'] == min_cluster_size and r['silhouette'] > -1]
        if samples_data:
            min_samples_vals = [r['min_samples'] for r in samples_data]
            silhouette_vals = [r['silhouette'] for r in samples_data]
            ax2.plot(min_samples_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'Silhouette (min_cluster_size={min_cluster_size})')

    ax2.set_xlabel('min_samples')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('HDBSCAN: SILHOUETTE SCORE\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Внешние метрики
    if best_ari:
        # Показываем ARI для разных min_cluster_size при min_samples=1
        ari_data = [r for r in results if r['min_samples'] == 1 and r['ari'] > -1]
        if ari_data:
            sizes = [r['min_cluster_size'] for r in ari_data]
            ari_vals = [r['ari'] for r in ari_data]
            ax3.plot(sizes, ari_vals, 'go-', linewidth=2, markersize=6, label='ARI')

    ax3.set_xlabel('min_cluster_size')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('HDBSCAN: ADJUSTED RAND INDEX\n(min_samples=1, ↑ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = []
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    test_sizes = [2, 5, 10, 15, 20]
    for min_cluster_size in test_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='cosine')
        start_time = time.time()
        clusterer.fit(X)
        times.append(time.time() - start_time)

    ax4.plot(test_sizes, times, color='purple', linestyle='-', marker='o', linewidth=2, markersize=6)
    ax4.set_xlabel('min_cluster_size')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('HDBSCAN: ВРЕМЯ ВЫПОЛНЕНИЯ\n(min_samples=1)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_hdbscan_internal_metrics(results, best_params):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    min_cluster_sizes = sorted(set(r['min_cluster_size'] for r in results))

    # График 1: Silhouette Score для разных min_cluster_size
    silhouette_data = []
    for min_cluster_size in min_cluster_sizes:
        size_data = [r for r in results if
                     r['min_cluster_size'] == min_cluster_size and r['min_samples'] == 1 and r['silhouette'] > -1]
        if size_data:
            silhouette_data.append((min_cluster_size, size_data[0]['silhouette']))

    if silhouette_data:
        sizes, silhouettes = zip(*silhouette_data)
        ax1.plot(sizes, silhouettes, 'bo-', linewidth=2, markersize=6)

        if best_params:
            ax1.axvline(x=best_params['min_cluster_size'], color='red', linestyle='--', alpha=0.7,
                        label=f'Лучшее: {best_params["min_cluster_size"]}')

    ax1.set_xlabel('min_cluster_size')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('HDBSCAN: SILHOUETTE SCORE\n(min_samples=1, ↑ лучше)')
    ax1.grid(True, alpha=0.3)
    if best_params:
        ax1.legend()

    # График 2: Количество кластеров
    cluster_data = []
    for min_cluster_size in min_cluster_sizes:
        size_data = [r for r in results if r['min_cluster_size'] == min_cluster_size and r['min_samples'] == 1]
        if size_data:
            cluster_data.append((min_cluster_size, size_data[0]['n_clusters']))

    if cluster_data:
        sizes, clusters = zip(*cluster_data)
        ax2.plot(sizes, clusters, 'go-', linewidth=2, markersize=6, label='Кластеры')

    ax2.set_xlabel('min_cluster_size')
    ax2.set_ylabel('Количество кластеров')
    ax2.set_title('HDBSCAN: КОЛИЧЕСТВО КЛАСТЕРОВ\n(min_samples=1)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Простая функция для быстрой кластеризации HDBSCAN
def simple_hdbscan_cluster(texts, min_cluster_size=5, min_samples=1):
    """
    Минимальная кластеризация текстов с HDBSCAN
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    # HDBSCAN кластеризация
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cosine',  # Косинусное расстояние для текстов
        cluster_selection_epsilon=0.0
    )
    labels = clusterer.fit_predict(X)

    # Анализ результатов
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # Вычисление метрик (без шума)
    non_noise_mask = labels != -1

    if n_clusters >= 2 and np.sum(non_noise_mask) >= 2:
        metrics = {
            'silhouette': silhouette_score(X_dense[non_noise_mask], labels[non_noise_mask]),
            'calinski_harabasz': calinski_harabasz_score(X_dense[non_noise_mask], labels[non_noise_mask]),
            'davies_bouldin': davies_bouldin_score(X_dense[non_noise_mask], labels[non_noise_mask]),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_persistence': clusterer.cluster_persistence_ if hasattr(clusterer,
                                                                             'cluster_persistence_') else None
        }
    else:
        metrics = {
            'silhouette': -1,
            'calinski_harabasz': -1,
            'davies_bouldin': -1,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_persistence': None
        }

    # Простой вывод
    print(f"📊 HDBSCAN кластеризация {len(texts)} текстов:")
    print(f"⚙️  ПАРАМЕТРЫ: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    print(f"🎯 РЕЗУЛЬТАТЫ:")
    print(f"   Количество кластеров: {n_clusters}")
    print(f"   Шумовых точек: {n_noise} ({n_noise / len(texts) * 100:.1f}%)")

    if metrics['silhouette'] > -1:
        print(f"   Silhouette Score: {metrics['silhouette']:.3f}")
        print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.3f}")
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")

        # Интерпретация Silhouette Score
        silhouette_val = metrics['silhouette']
        if silhouette_val > 0.7:
            interpretation = "Отличное разделение"
        elif silhouette_val > 0.5:
            interpretation = "Разумное разделение"
        elif silhouette_val > 0.25:
            interpretation = "Слабое разделение"
        else:
            interpretation = "Нет существенного разделения"
        print(f"   Интерпретация: {interpretation}")
    else:
        print("   ⚠️  Недостаточно кластеров для вычисления метрик")

    # Информация о кластерах
    print(f"\n🔍 ИНФОРМАЦИЯ О КЛАСТЕРАХ:")
    for label in sorted(unique_labels):
        if label == -1:
            print(f"🔸 Шум: {n_noise} текстов")
            continue

        cluster_texts = [texts[j] for j, lbl in enumerate(labels) if lbl == label]
        print(f"🔸 Кластер {label}: {len(cluster_texts)} текстов")
        if len(cluster_texts) > 0:
            for text in cluster_texts[:2]:
                print(f"   - {text[:60]}..." if len(text) > 60 else f"   - {text}")
            if len(cluster_texts) > 2:
                print(f"   ... и еще {len(cluster_texts) - 2} текстов")
        print()

    return labels, metrics, clusterer


# Функция для визуализации дерева кластеров HDBSCAN
def plot_cluster_tree(clusterer, texts):
    """
    Визуализация дерева кластеров HDBSCAN
    """
    if hasattr(clusterer, 'condensed_tree_'):
        plt.figure(figsize=(10, 6))
        clusterer.condensed_tree_.plot(select_clusters=True,
                                       selection_palette=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('HDBSCAN: ДЕРЕВО КЛАСТЕРОВ')
        plt.tight_layout()
        plt.show()

        # Дополнительная информация о кластерах
        if hasattr(clusterer, 'cluster_persistence_'):
            print("\n📊 ИНФОРМАЦИЯ О УСТОЙЧИВОСТИ КЛАСТЕРОВ:")
            for i, persistence in enumerate(clusterer.cluster_persistence_):
                print(f"   Кластер {i}: устойчивость = {persistence:.3f}")


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 HDBSCAN ДЛЯ ТЕКСТОВ")
    print("=" * 50)

    # Установка hdbscan если не установлен
    try:
        import hdbscan
    except ImportError:
        print("❌ HDBSCAN не установлен. Установите: pip install hdbscan")
        exit()

    # Шаг 1: Сравнение с внешними метриками
    print("🎯 ШАГ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_int, best_by_ext = compare_hdbscan_parameters(
        texts, true_labels,
        min_cluster_size_range=[2, 3, 5, 10, 15],
        min_samples_range=[1, 2, 3]
    )

    # Шаг 2: Быстрая кластеризация
    print("\n🎯 ШАГ 2: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ")
    if best_by_int:
        labels, metrics, clusterer = simple_hdbscan_cluster(
            texts,
            min_cluster_size=best_by_int['min_cluster_size'],
            min_samples=best_by_int['min_samples']
        )
    else:
        # Используем параметры по умолчанию если не нашли лучших
        labels, metrics, clusterer = simple_hdbscan_cluster(
            texts,
            min_cluster_size=5,
            min_samples=1
        )

    # Шаг 3: Визуализация дерева кластеров
    print("\n🎯 ШАГ 3: ВИЗУАЛИЗАЦИЯ ДЕРЕВА КЛАСТЕРОВ")
    plot_cluster_tree(clusterer, texts)