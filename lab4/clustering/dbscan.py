import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score)
from sklearn.neighbors import NearestNeighbors

from util.decribe import get_labels, get_texts


def compare_dbscan_parameters(texts, true_labels=None, max_eps=0.5, eps_step=0.1, min_samples_range=None):
    """
    Сравнение DBSCAN с различными параметрами eps и min_samples
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    has_true_labels = true_labels is not None

    if min_samples_range is None:
        min_samples_range = [2, 3, 5]

    if has_true_labels:
        print("🔬 DBSCAN С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("eps\tmin_sam\tClusters\tNoise\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
        print("-" * 100)
    else:
        print("🔬 DBSCAN С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("eps\tmin_sam\tClusters\tNoise\tSilhouette\tCalinski\tDavies-B")
        print("-" * 70)

    results = []

    for eps in np.arange(0.1, max_eps + eps_step, eps_step):
        for min_samples in min_samples_range:
            # DBSCAN кластеризация
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'  # Используем косинусное расстояние для текстов
            )
            labels = dbscan.fit_predict(X)

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

                print(f"{eps:.1f}\t{min_samples}\t{n_clusters}\t\t{n_noise}\t{silhouette:.3f}\t\t"
                      f"{calinski:.3f}\t\t{davies:.3f}\t\t{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
            else:
                print(f"{eps:.1f}\t{min_samples}\t{n_clusters}\t\t{n_noise}\t{silhouette:.3f}\t\t"
                      f"{calinski:.3f}\t\t{davies:.3f}")

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'ari': ari if has_true_labels else -1,
                'nmi': nmi if has_true_labels else -1,
                'v_measure': v_measure if has_true_labels else -1,
                'labels': labels
            })

    # Находим лучшие параметры
    valid_results = [r for r in results if r['n_clusters'] >= 2]

    if valid_results:
        best_by_silhouette = max(valid_results, key=lambda x: x['silhouette'])
        if has_true_labels:
            best_by_ari = max(valid_results, key=lambda x: x['ari'])

        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print(f"   По внутренним метрикам: eps={best_by_silhouette['eps']:.1f}, "
              f"min_samples={best_by_silhouette['min_samples']} "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f}, "
              f"Кластеров: {best_by_silhouette['n_clusters']})")
        if has_true_labels:
            print(f"   По внешним метрикам: eps={best_by_ari['eps']:.1f}, "
                  f"min_samples={best_by_ari['min_samples']} "
                  f"(ARI: {best_by_ari['ari']:.3f}, "
                  f"Кластеров: {best_by_ari['n_clusters']})")
    else:
        print(f"\n⚠️  Не найдено параметров, создающих хотя бы 2 кластера")
        best_by_silhouette = best_by_ari = None

    # Строим графики
    if has_true_labels and valid_results:
        _plot_dbscan_metrics(results, texts, true_labels, best_by_silhouette, best_by_ari)
    elif valid_results:
        _plot_dbscan_internal_metrics(results, best_by_silhouette)

    if has_true_labels and valid_results:
        return best_by_silhouette, best_by_ari
    elif valid_results:
        return best_by_silhouette
    else:
        return None, None if has_true_labels else None


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
    plt.xlabel('Точки')
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


def _plot_dbscan_metrics(results, texts, true_labels, best_silhouette, best_ari):
    """
    Построение графиков для DBSCAN со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Подготовка данных для графиков
    eps_values = sorted(set(r['eps'] for r in results))
    min_samples_values = sorted(set(r['min_samples'] for r in results))

    # График 1: Количество кластеров и шума
    for min_samples in min_samples_values:
        cluster_data = [r for r in results if r['min_samples'] == min_samples]
        eps_vals = [r['eps'] for r in cluster_data]
        cluster_vals = [r['n_clusters'] for r in cluster_data]
        noise_vals = [r['n_noise'] for r in cluster_data]

        ax1.plot(eps_vals, cluster_vals, 'o-', linewidth=2,
                 label=f'Кластеры (min_samples={min_samples})')
        ax1.plot(eps_vals, noise_vals, 'o--', linewidth=2,
                 label=f'Шум (min_samples={min_samples})')

    if best_silhouette:
        ax1.axvline(x=best_silhouette['eps'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Лучшее eps={best_silhouette["eps"]:.1f} (внутр.)')
    if best_ari:
        ax1.axvline(x=best_ari['eps'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее eps={best_ari["eps"]:.1f} (внеш.)')

    ax1.set_xlabel('Eps')
    ax1.set_ylabel('Количество')
    ax1.set_title('DBSCAN: КЛАСТЕРЫ И ШУМ\n(↑ кластеры, ↓ шум)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Внутренние метрики
    for min_samples in min_samples_values:
        cluster_data = [r for r in results if r['min_samples'] == min_samples and r['silhouette'] > -1]
        if cluster_data:
            eps_vals = [r['eps'] for r in cluster_data]
            silhouette_vals = [r['silhouette'] for r in cluster_data]
            ax2.plot(eps_vals, silhouette_vals, 'o-', linewidth=2,
                     label=f'Silhouette (min_samples={min_samples})')

    ax2.set_xlabel('Eps')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('DBSCAN: SILHOUETTE SCORE\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Внешние метрики
    for min_samples in min_samples_values:
        cluster_data = [r for r in results if r['min_samples'] == min_samples and r['ari'] > -1]
        if cluster_data:
            eps_vals = [r['eps'] for r in cluster_data]
            ari_vals = [r['ari'] for r in cluster_data]
            ax3.plot(eps_vals, ari_vals, 'o-', linewidth=2,
                     label=f'ARI (min_samples={min_samples})')

    ax3.set_xlabel('Eps')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('DBSCAN: ADJUSTED RAND INDEX\n(↑ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = []
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    test_eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    for eps in test_eps:
        dbscan = DBSCAN(eps=eps, min_samples=3, metric='cosine')
        start_time = time.time()
        dbscan.fit(X)
        times.append(time.time() - start_time)

    ax4.plot(test_eps, times, color='purple', linestyle='-', marker='o', linewidth=2)
    ax4.set_xlabel('Eps')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('DBSCAN: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_dbscan_internal_metrics(results, best_params):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    min_samples_values = sorted(set(r['min_samples'] for r in results))

    # График 1: Silhouette Score
    for min_samples in min_samples_values:
        cluster_data = [r for r in results if r['min_samples'] == min_samples and r['silhouette'] > -1]
        if cluster_data:
            eps_vals = [r['eps'] for r in cluster_data]
            silhouette_vals = [r['silhouette'] for r in cluster_data]
            ax1.plot(eps_vals, silhouette_vals, 'o-', linewidth=2,
                     label=f'min_samples={min_samples}')

    if best_params:
        ax1.axvline(x=best_params['eps'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее eps={best_params["eps"]:.1f}')

    ax1.set_xlabel('Eps')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('DBSCAN: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Количество кластеров
    for min_samples in min_samples_values:
        cluster_data = [r for r in results if r['min_samples'] == min_samples]
        eps_vals = [r['eps'] for r in cluster_data]
        cluster_vals = [r['n_clusters'] for r in cluster_data]
        ax2.plot(eps_vals, cluster_vals, 'o-', linewidth=2,
                 label=f'min_samples={min_samples}')

    ax2.set_xlabel('Eps')
    ax2.set_ylabel('Количество кластеров')
    ax2.set_title('DBSCAN: КОЛИЧЕСТВО КЛАСТЕРОВ')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Простая функция для быстрой кластеризации DBSCAN
def simple_dbscan_cluster(texts, eps=0.3, min_samples=3):
    """
    Минимальная кластеризация текстов с DBSCAN
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    # DBSCAN кластеризация
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine'  # Косинусное расстояние для текстов
    )
    labels = dbscan.fit_predict(X)

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
            'n_noise': n_noise
        }
    else:
        metrics = {
            'silhouette': -1,
            'calinski_harabasz': -1,
            'davies_bouldin': -1,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }

    # Простой вывод
    print(f"📊 DBSCAN кластеризация {len(texts)} текстов:")
    print(f"⚙️  ПАРАМЕТРЫ: eps={eps}, min_samples={min_samples}")
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

    return labels, metrics


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 DBSCAN ДЛЯ ТЕКСТОВ")
    print("=" * 50)

    # Шаг 1: Поиск оптимального eps
    print("🎯 ШАГ 1: ПОИСК ОПТИМАЛЬНОГО EPS")
    recommended_eps = find_optimal_eps(texts, k=5)

    # Шаг 2: Сравнение с внешними метриками
    print("\n🎯 ШАГ 2: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_int, best_by_ext = compare_dbscan_parameters(
        texts, true_labels, max_eps=0.5, eps_step=0.1, min_samples_range=[2, 3, 5]
    )

    # Шаг 3: Быстрая кластеризация
    print("\n🎯 ШАГ 3: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ")
    if best_by_int:
        labels, metrics = simple_dbscan_cluster(
            texts,
            eps=best_by_int['eps'],
            min_samples=best_by_int['min_samples']
        )
    else:
        labels, metrics = simple_dbscan_cluster(
            texts,
            eps=recommended_eps,
            min_samples=3
        )