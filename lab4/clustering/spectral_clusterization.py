from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time

from util.decribe import get_labels, get_texts


def compare_spectral_clustering(texts, true_labels=None, max_k=6, affinity_types=None, n_neighbors_range=None):
    """
    Сравнение SpectralClustering с разными типами сходства и параметрами
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    has_true_labels = true_labels is not None

    if affinity_types is None:
        affinity_types = ['rbf', 'nearest_neighbors', 'cosine']
    if n_neighbors_range is None:
        n_neighbors_range = [5, 10, 15]

    if has_true_labels:
        print("🔬 SPECTRAL CLUSTERING С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("k\tAffinity\tn_neigh\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
        print("-" * 105)
    else:
        print("🔬 SPECTRAL CLUSTERING С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("k\tAffinity\tn_neigh\tSilhouette\tCalinski\tDavies-B")
        print("-" * 75)

    results = []

    for k in range(2, max_k + 1):
        for affinity in affinity_types:
            # Для nearest_neighbors тестируем разные n_neighbors
            if affinity == 'nearest_neighbors':
                n_neighbors_list = n_neighbors_range
            else:
                n_neighbors_list = [None]  # Для других affinity n_neighbors не используется

            for n_neighbors in n_neighbors_list:
                try:
                    # SpectralClustering кластеризация
                    spectral = SpectralClustering(
                        n_clusters=k,
                        affinity=affinity,
                        n_neighbors=n_neighbors,
                        random_state=42,
                        n_init=10
                    )

                    # Для cosine affinity используем разреженную матрицу, для других - плотную
                    if affinity == 'cosine':
                        X_used = X
                    else:
                        X_used = X_dense

                    labels = spectral.fit_predict(X_used)

                    # Внутренние метрики
                    silhouette = silhouette_score(X_dense, labels)
                    calinski = calinski_harabasz_score(X_dense, labels)
                    davies = davies_bouldin_score(X_dense, labels)

                    if has_true_labels:
                        # Внешние метрики
                        ari = adjusted_rand_score(true_labels, labels)
                        nmi = normalized_mutual_info_score(true_labels, labels)
                        v_measure = v_measure_score(true_labels, labels)

                        n_neigh_str = str(n_neighbors) if n_neighbors else "N/A"
                        print(f"{k}\t{affinity}\t{n_neigh_str}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
                              f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
                    else:
                        n_neigh_str = str(n_neighbors) if n_neighbors else "N/A"
                        print(f"{k}\t{affinity}\t{n_neigh_str}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}")

                    results.append({
                        'k': k,
                        'affinity': affinity,
                        'n_neighbors': n_neighbors,
                        'silhouette': silhouette,
                        'calinski': calinski,
                        'davies': davies,
                        'ari': ari if has_true_labels else -1,
                        'nmi': nmi if has_true_labels else -1,
                        'v_measure': v_measure if has_true_labels else -1,
                        'labels': labels
                    })

                except Exception as e:
                    n_neigh_str = str(n_neighbors) if n_neighbors else "N/A"
                    print(f"{k}\t{affinity}\t{n_neigh_str}\tERROR: {str(e)[:30]}...")

    # Находим лучшие параметры
    if results:
        best_by_silhouette = max(results, key=lambda x: x['silhouette'])
        if has_true_labels:
            best_by_ari = max(results, key=lambda x: x['ari'])

        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print(f"   По внутренним метрикам: k={best_by_silhouette['k']}, "
              f"affinity={best_by_silhouette['affinity']}, "
              f"n_neighbors={best_by_silhouette['n_neighbors']} "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f})")
        if has_true_labels:
            print(f"   По внешним метрикам: k={best_by_ari['k']}, "
                  f"affinity={best_by_ari['affinity']}, "
                  f"n_neighbors={best_by_ari['n_neighbors']} "
                  f"(ARI: {best_by_ari['ari']:.3f})")
    else:
        print(f"\n⚠️  Не удалось выполнить кластеризацию")
        best_by_silhouette = best_by_ari = None

    # Строим графики
    if has_true_labels and results:
        _plot_spectral_metrics(results, texts, true_labels, best_by_silhouette, best_by_ari)
    elif results:
        _plot_spectral_internal_metrics(results, best_by_silhouette)

    if has_true_labels and results:
        return best_by_silhouette, best_by_ari
    elif results:
        return best_by_silhouette
    else:
        return None, None


def _plot_spectral_metrics(results, texts, true_labels, best_silhouette, best_ari):
    """
    Построение графиков для SpectralClustering со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Подготовка данных для графиков
    affinity_types = sorted(set(r['affinity'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score для разных affinity (n_neighbors=10 для nearest_neighbors)
    for affinity in affinity_types:
        if affinity == 'nearest_neighbors':
            # Берем только n_neighbors=10 для сравнения
            aff_data = [r for r in results if r['affinity'] == affinity and r['n_neighbors'] == 10]
        else:
            aff_data = [r for r in results if r['affinity'] == affinity]

        if aff_data:
            k_vals = [r['k'] for r in aff_data]
            silhouette_vals = [r['silhouette'] for r in aff_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{affinity}')

    if best_silhouette:
        ax1.axvline(x=best_silhouette['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_silhouette["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('SPECTRAL: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Влияние n_neighbors для nearest_neighbors affinity
    n_neighbors_vals = sorted(set(r['n_neighbors'] for r in results if r['n_neighbors'] is not None))
    k_for_plot = 3  # Выбираем одно k для демонстрации

    for n_neighbors in n_neighbors_vals:
        nn_data = [r for r in results if r['n_neighbors'] == n_neighbors and r['affinity'] == 'nearest_neighbors']
        if nn_data:
            k_vals = [r['k'] for r in nn_data]
            silhouette_vals = [r['silhouette'] for r in nn_data]
            ax2.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'n_neighbors={n_neighbors}')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('SPECTRAL: ВЛИЯНИЕ n_neighbors\n(nearest_neighbors affinity)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Внешние метрики (ARI)
    if best_ari:
        for affinity in affinity_types:
            if affinity == 'nearest_neighbors':
                aff_data = [r for r in results if
                            r['affinity'] == affinity and r['n_neighbors'] == 10 and r['ari'] > -1]
            else:
                aff_data = [r for r in results if r['affinity'] == affinity and r['ari'] > -1]

            if aff_data:
                k_vals = [r['k'] for r in aff_data]
                ari_vals = [r['ari'] for r in aff_data]
                ax3.plot(k_vals, ari_vals, 'o-', linewidth=2, markersize=6,
                         label=f'{affinity}')

    ax3.set_xlabel('Количество кластеров (k)')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('SPECTRAL: ADJUSTED RAND INDEX\n(↑ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = {}
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    for affinity in affinity_types:
        aff_times = []
        for k in k_values:
            try:
                if affinity == 'nearest_neighbors':
                    spectral = SpectralClustering(n_clusters=k, affinity=affinity,
                                                  n_neighbors=10, random_state=42)
                else:
                    spectral = SpectralClustering(n_clusters=k, affinity=affinity,
                                                  random_state=42)

                # Выбираем правильный формат данных
                X_used = X if affinity == 'cosine' else X_dense

                start_time = time.time()
                spectral.fit(X_used)
                aff_times.append(time.time() - start_time)
            except:
                aff_times.append(np.nan)

        times[affinity] = aff_times

    for affinity, time_vals in times.items():
        ax4.plot(k_values, time_vals, 'o-', linewidth=2, markersize=6, label=affinity)

    ax4.set_xlabel('Количество кластеров (k)')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('SPECTRAL: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def _plot_spectral_internal_metrics(results, best_params):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    affinity_types = sorted(set(r['affinity'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score для разных affinity
    for affinity in affinity_types:
        if affinity == 'nearest_neighbors':
            # Берем n_neighbors=10 для сравнения
            aff_data = [r for r in results if r['affinity'] == affinity and r['n_neighbors'] == 10]
        else:
            aff_data = [r for r in results if r['affinity'] == affinity]

        if aff_data:
            k_vals = [r['k'] for r in aff_data]
            silhouette_vals = [r['silhouette'] for r in aff_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{affinity}')

    if best_params:
        ax1.axvline(x=best_params['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_params["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('SPECTRAL: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Calinski-Harabasz Score
    for affinity in affinity_types:
        if affinity == 'nearest_neighbors':
            aff_data = [r for r in results if r['affinity'] == affinity and r['n_neighbors'] == 10]
        else:
            aff_data = [r for r in results if r['affinity'] == affinity]

        if aff_data:
            k_vals = [r['k'] for r in aff_data]
            calinski_vals = [r['calinski'] for r in aff_data]
            ax2.plot(k_vals, calinski_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{affinity}')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.set_title('SPECTRAL: CALINSKI-HARABASZ\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Простая функция для быстрой кластеризации SpectralClustering
def simple_spectral_cluster(texts, n_clusters=3, affinity='cosine', n_neighbors=10):
    """
    Минимальная кластеризация текстов с SpectralClustering
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    # SpectralClustering кластеризация
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors if affinity == 'nearest_neighbors' else None,
        random_state=42,
        n_init=10
    )

    # Выбираем правильный формат данных
    if affinity == 'cosine':
        X_used = X
    else:
        X_used = X_dense

    labels = spectral.fit_predict(X_used)

    # Вычисление метрик
    metrics = {
        'silhouette': silhouette_score(X_dense, labels),
        'calinski_harabasz': calinski_harabasz_score(X_dense, labels),
        'davies_bouldin': davies_bouldin_score(X_dense, labels),
        'affinity': affinity,
        'n_neighbors': n_neighbors if affinity == 'nearest_neighbors' else 'N/A'
    }

    # Простой вывод
    print(f"📊 SpectralClustering кластеризация {len(texts)} текстов на {n_clusters} кластеров:")
    print(f"⚙️  ПАРАМЕТРЫ: affinity={affinity}, n_neighbors={metrics['n_neighbors']}")
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

    # Информация о типах сходства
    print(f"\n📋 ТИПЫ СХОДСТВА (AFFINITY):")
    affinity_info = {
        'rbf': "Радиальная базисная функция (гауссово ядро) - евклидово расстояние",
        'nearest_neighbors': "Граф ближайших соседей - локальная структура данных",
        'cosine': "Косинусное сходство - хорошо для текстовых данных"
    }
    print(f"   {affinity}: {affinity_info.get(affinity, '')}")

    # Особенности Spectral Clustering
    print(f"\n💡 ОСОБЕННОСТИ SPECTRAL CLUSTERING:")
    print("   • Работает с данными сложной формы")
    print("   • Использует спектральную теорию графов")
    print("   • Эффективен когда кластеры не являются выпуклыми")
    print("   • Чувствителен к выбору параметров сходства")
    print("   • Вычислительно сложнее чем K-means")

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


# Функция для анализа графа сходства
def analyze_affinity_graph(texts, affinity='cosine', n_neighbors=10):
    """
    Анализ графа сходства для Spectral Clustering
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    print(f"🔗 АНАЛИЗ ГРАФА СХОДСТВА:")
    print(f"   Тип сходства: {affinity}")

    if affinity == 'nearest_neighbors':
        # Строим граф ближайших соседей
        connectivity = kneighbors_graph(X_dense, n_neighbors=n_neighbors, include_self=False)
        n_components = connectivity.shape[0]
        n_edges = connectivity.nnz // 2  # Неориентированный граф

        print(f"   Количество вершин: {n_components}")
        print(f"   Количество рёбер: {n_edges}")
        print(f"   Плотность графа: {n_edges / (n_components * (n_components - 1) / 2):.4f}")

    elif affinity == 'cosine':
        # Для косинусного сходства вычисляем среднее сходство
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(X)
        np.fill_diagonal(similarities, 0)  # Игнорируем самосходство
        avg_similarity = np.mean(similarities)

        print(f"   Среднее косинусное сходство: {avg_similarity:.4f}")
        print(f"   Максимальное сходство: {np.max(similarities):.4f}")
        print(f"   Минимальное сходство: {np.min(similarities):.4f}")

    return True


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 SPECTRAL CLUSTERING ДЛЯ ТЕКСТОВ")
    print("=" * 60)

    # Анализ графа сходства
    print("🎯 АНАЛИЗ СТРУКТУРЫ ДАННЫХ:")
    analyze_affinity_graph(texts, affinity='cosine')

    # Вариант 1: Сравнение с внешними метриками
    print("\n🎯 ВАРИАНТ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_int, best_by_ext = compare_spectral_clustering(
        texts, true_labels, max_k=5,
        affinity_types=['rbf', 'nearest_neighbors', 'cosine'],
        n_neighbors_range=[5, 10, 15]
    )

    # Вариант 2: Быстрая кластеризация
    print("\n🎯 ВАРИАНТ 2: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ")
    if best_by_int:
        labels, metrics = simple_spectral_cluster(
            texts,
            n_clusters=best_by_int['k'],
            affinity=best_by_int['affinity'],
            n_neighbors=best_by_int['n_neighbors'] if best_by_int['affinity'] == 'nearest_neighbors' else 10
        )
    else:
        labels, metrics = simple_spectral_cluster(
            texts,
            n_clusters=3,
            affinity='cosine'
        )