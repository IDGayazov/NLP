import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score)
from scipy.sparse import issparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from util.decribe import get_labels, get_texts


def compare_agglomerative_clustering(texts, true_labels=None, max_k=6, linkage_types=None):
    """
    Сравнение AgglomerativeClustering с разными типами связей
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    # Для внутренних метрик нужны плотные данные
    X_dense = X.toarray() if issparse(X) else X

    has_true_labels = true_labels is not None

    if linkage_types is None:
        linkage_types = ['ward', 'average', 'complete', 'single']

    if has_true_labels:
        print("🔬 AGGLOMERATIVE CLUSTERING С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("k\tLinkage\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
        print("-" * 95)
    else:
        print("🔬 AGGLOMERATIVE CLUSTERING С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("k\tLinkage\tSilhouette\tCalinski\tDavies-B")
        print("-" * 65)

    results = []

    for k in range(2, max_k + 1):
        for linkage_type in linkage_types:
            try:
                start_time = time.time()

                if linkage_type == 'ward':
                    # Используем sklearn для ward (работает с плотными данными)
                    agglo = AgglomerativeClustering(
                        n_clusters=k,
                        linkage='ward',
                        metric='euclidean'
                    )
                    labels = agglo.fit_predict(X_dense)
                else:
                    # Для других linkage используем scipy.hierarchy
                    # Вычисляем попарные расстояния
                    if linkage_type in ['average', 'complete', 'single']:
                        # Используем косинусное расстояние для текстов
                        distances = pdist(X_dense, metric='cosine')
                    else:
                        distances = pdist(X_dense, metric='euclidean')

                    # Строим иерархию
                    Z = linkage(distances, method=linkage_type)
                    # Получаем кластеры
                    labels = fcluster(Z, k, criterion='maxclust') - 1  # Приводим к 0-based

                fit_time = time.time() - start_time

                # Внутренние метрики всегда на плотных данных
                silhouette = silhouette_score(X_dense, labels)
                calinski = calinski_harabasz_score(X_dense, labels)
                davies = davies_bouldin_score(X_dense, labels)

                if has_true_labels:
                    # Внешние метрики
                    ari = adjusted_rand_score(true_labels, labels)
                    nmi = normalized_mutual_info_score(true_labels, labels)
                    v_measure = v_measure_score(true_labels, labels)

                    print(f"{k}\t{linkage_type}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
                          f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
                else:
                    print(f"{k}\t{linkage_type}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}")

                results.append({
                    'k': k,
                    'linkage': linkage_type,
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies,
                    'ari': ari if has_true_labels else -1,
                    'nmi': nmi if has_true_labels else -1,
                    'v_measure': v_measure if has_true_labels else -1,
                    'labels': labels,
                    'fit_time': fit_time
                })

            except Exception as e:
                error_msg = str(e)
                print(f"{k}\t{linkage_type}\tERROR: {error_msg[:40]}...")

    # Находим лучшие параметры
    if results:
        best_by_silhouette = max(results, key=lambda x: x['silhouette'])
        if has_true_labels:
            best_by_ari = max(results, key=lambda x: x['ari'])

        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print(f"   По внутренним метрикам: k={best_by_silhouette['k']}, "
              f"linkage={best_by_silhouette['linkage']} "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f})")
        if has_true_labels:
            print(f"   По внешним метрикам: k={best_by_ari['k']}, "
                  f"linkage={best_by_ari['linkage']} "
                  f"(ARI: {best_by_ari['ari']:.3f})")
    else:
        print(f"\n⚠️  Не удалось выполнить кластеризацию")
        best_by_silhouette = best_by_ari = None

    # Строим графики
    if has_true_labels and results:
        _plot_agglomerative_metrics(results, texts, true_labels, best_by_silhouette, best_by_ari)
    elif results:
        _plot_agglomerative_internal_metrics(results, best_by_silhouette)

    if has_true_labels and results:
        return best_by_silhouette, best_by_ari
    elif results:
        return best_by_silhouette
    else:
        return None, None if has_true_labels else None


def _plot_agglomerative_metrics(results, texts, true_labels, best_silhouette, best_ari):
    """
    Построение графиков для AgglomerativeClustering со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Подготовка данных для графиков
    linkage_types = sorted(set(r['linkage'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score для разных linkage
    for linkage_type in linkage_types:
        linkage_data = [r for r in results if r['linkage'] == linkage_type]
        if linkage_data:
            k_vals = [r['k'] for r in linkage_data]
            silhouette_vals = [r['silhouette'] for r in linkage_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{linkage_type}')

    if best_silhouette:
        ax1.axvline(x=best_silhouette['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_silhouette["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('AGGLOMERATIVE: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Внешние метрики (ARI)
    for linkage_type in linkage_types:
        linkage_data = [r for r in results if r['linkage'] == linkage_type and r['ari'] > -1]
        if linkage_data:
            k_vals = [r['k'] for r in linkage_data]
            ari_vals = [r['ari'] for r in linkage_data]
            ax2.plot(k_vals, ari_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{linkage_type}')

    if best_ari:
        ax2.axvline(x=best_ari['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_ari["k"]}')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('ARI Score')
    ax2.set_title('AGGLOMERATIVE: ADJUSTED RAND INDEX\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Сравнение метрик для лучшего linkage
    if results:
        # Находим лучший linkage по silhouette
        best_linkage_data = {}
        for linkage_type in linkage_types:
            linkage_results = [r for r in results if r['linkage'] == linkage_type]
            if linkage_results:
                best_for_linkage = max(linkage_results, key=lambda x: x['silhouette'])
                best_linkage_data[linkage_type] = best_for_linkage

        if best_linkage_data:
            linkages = list(best_linkage_data.keys())
            silhouettes = [best_linkage_data[linkage]['silhouette'] for linkage in linkages]
            aris = [best_linkage_data[linkage]['ari'] for linkage in linkages] if best_ari else [0] * len(linkages)

            x = np.arange(len(linkages))
            width = 0.35

            ax3.bar(x - width / 2, silhouettes, width, label='Silhouette', alpha=0.8)
            if best_ari:
                ax3.bar(x + width / 2, aris, width, label='ARI', alpha=0.8)

            ax3.set_xlabel('Linkage Type')
            ax3.set_ylabel('Score')
            ax3.set_title('СРАВНЕНИЕ LINKAGE ТИПОВ\n(лучшие результаты для каждого k)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(linkages)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    # График 4: Время выполнения
    times = {}
    for linkage_type in linkage_types:
        linkage_times = []
        for k in k_values:
            linkage_results = [r for r in results if r['linkage'] == linkage_type and r['k'] == k]
            if linkage_results:
                linkage_times.append(linkage_results[0]['fit_time'])
        if linkage_times:
            times[linkage_type] = linkage_times

    for linkage_type, time_vals in times.items():
        if len(time_vals) == len(k_values):
            ax4.plot(k_values, time_vals, 'o-', linewidth=2, markersize=6, label=linkage_type)

    ax4.set_xlabel('Количество кластеров (k)')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('AGGLOMERATIVE: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def _plot_agglomerative_internal_metrics(results, best_params):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    linkage_types = sorted(set(r['linkage'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score
    for linkage_type in linkage_types:
        linkage_data = [r for r in results if r['linkage'] == linkage_type]
        if linkage_data:
            k_vals = [r['k'] for r in linkage_data]
            silhouette_vals = [r['silhouette'] for r in linkage_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{linkage_type}')

    if best_params:
        ax1.axvline(x=best_params['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_params["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('AGGLOMERATIVE: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Calinski-Harabasz Score
    for linkage_type in linkage_types:
        linkage_data = [r for r in results if r['linkage'] == linkage_type]
        if linkage_data:
            k_vals = [r['k'] for r in linkage_data]
            calinski_vals = [r['calinski'] for r in linkage_data]
            ax2.plot(k_vals, calinski_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{linkage_type}')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.set_title('AGGLOMERATIVE: CALINSKI-HARABASZ\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Davies-Bouldin Score
    for linkage_type in linkage_types:
        linkage_data = [r for r in results if r['linkage'] == linkage_type]
        if linkage_data:
            k_vals = [r['k'] for r in linkage_data]
            davies_vals = [r['davies'] for r in linkage_data]
            ax3.plot(k_vals, davies_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{linkage_type}')

    ax3.set_xlabel('Количество кластеров (k)')
    ax3.set_ylabel('Davies-Bouldin Score')
    ax3.set_title('AGGLOMERATIVE: DAVIES-BOULDIN\n(↓ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def simple_agglomerative_cluster(texts, n_clusters=3, linkage_type='ward'):
    """
    Минимальная кластеризация текстов с AgglomerativeClustering
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray() if issparse(X) else X

    start_time = time.time()

    if linkage_type == 'ward':
        # Используем sklearn для ward
        agglo = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        labels = agglo.fit_predict(X_dense)
    else:
        # Для других linkage используем scipy.hierarchy
        if linkage_type in ['average', 'complete', 'single']:
            distances = pdist(X_dense, metric='cosine')
        else:
            distances = pdist(X_dense, metric='euclidean')

        Z = linkage(distances, method=linkage_type)
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

    fit_time = time.time() - start_time

    # Вычисление метрик
    metrics = {
        'silhouette': silhouette_score(X_dense, labels),
        'calinski_harabasz': calinski_harabasz_score(X_dense, labels),
        'davies_bouldin': davies_bouldin_score(X_dense, labels),
        'linkage': linkage_type,
        'fit_time': fit_time
    }

    # Простой вывод
    print(f"📊 AgglomerativeClustering кластеризация {len(texts)} текстов на {n_clusters} кластеров:")
    print(f"⚙️  ПАРАМЕТРЫ: linkage={linkage_type}")
    print(f"⏱️  Время выполнения: {fit_time:.2f} сек")
    print(f"🎯 МЕТРИКИ:")
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

    # Информация о типах связей
    print(f"\n📋 ОСОБЕННОСТИ LINKAGE ТИПОВ:")
    linkage_info = {
        'ward': "Минимизирует дисперсию внутри кластеров (евклидова метрика)",
        'average': "Среднее расстояние между всеми точками кластеров (косинусная метрика)",
        'complete': "Максимальное расстояние между точками кластеров (косинусная метрика)",
        'single': "Минимальное расстояние между точками кластеров (косинусная метрика)"
    }
    print(f"   {linkage_type}: {linkage_info.get(linkage_type, '')}")

    # Информация о кластерах
    print(f"\n🔍 ИНФОРМАЦИЯ О КЛАСТЕРАХ:")
    unique_labels = np.unique(labels)
    for i in unique_labels:
        cluster_texts = [texts[j] for j, label in enumerate(labels) if label == i]
        print(f"🔸 Кластер {i}: {len(cluster_texts)} текстов")
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

    print("🚀 AGGLOMERATIVE CLUSTERING ДЛЯ ТЕКСТОВ")
    print("=" * 60)

    # Вариант 1: Сравнение с внешними метриками
    print("🎯 ВАРИАНТ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_int, best_by_ext = compare_agglomerative_clustering(
        texts, true_labels, max_k=5,
        linkage_types=['ward', 'average', 'complete', 'single']
    )

    # Вариант 2: Быстрая кластеризация
    print("\n🎯 ВАРИАНТ 2: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ")
    if best_by_int:
        labels, metrics = simple_agglomerative_cluster(
            texts,
            n_clusters=best_by_int['k'],
            linkage_type=best_by_int['linkage']
        )
    else:
        # Используем параметры по умолчанию если не нашли лучших
        labels, metrics = simple_agglomerative_cluster(
            texts,
            n_clusters=3,
            linkage_type='average'
        )