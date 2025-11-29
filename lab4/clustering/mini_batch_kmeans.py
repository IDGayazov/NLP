from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
import matplotlib.pyplot as plt
import numpy as np
import time

from util.decribe import get_labels, get_texts


def compare_minibatch_kmeans_sizes(texts, true_labels=None, max_k=6, batch_size=100):
    """
    Сравнение MiniBatchKMeans с внутренними и внешними метриками
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    has_true_labels = true_labels is not None

    if has_true_labels:
        print("🔬 MINIBATCHKMEANS С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("k\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
        print("-" * 85)
    else:
        print("🔬 MINIBATCHKMEANS С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("k\tSilhouette\tCalinski\tDavies-B")
        print("-" * 55)

    # Списки для хранения метрик
    k_values = []
    internal_metrics = {'silhouette': [], 'calinski': [], 'davies': []}
    external_metrics = {'ari': [], 'nmi': [], 'v_measure': []}

    best_k_internal = 2
    best_k_external = 2
    best_silhouette = -1
    best_ari = -1

    for k in range(2, max_k + 1):
        # MiniBatchKMeans с настройками для текстовых данных
        mbkmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            random_state=42,
            n_init=3,  # Меньше инициализаций для скорости
            max_iter=100
        )
        labels = mbkmeans.fit_predict(X)

        # Внутренние метрики
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        if has_true_labels:
            # Внешние метрики
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)
            v_measure = v_measure_score(true_labels, labels)

            print(f"{k}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
                  f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
        else:
            print(f"{k}\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}")

        # Сохраняем метрики
        k_values.append(k)
        internal_metrics['silhouette'].append(silhouette)
        internal_metrics['calinski'].append(calinski)
        internal_metrics['davies'].append(davies)

        if has_true_labels:
            external_metrics['ari'].append(ari)
            external_metrics['nmi'].append(nmi)
            external_metrics['v_measure'].append(v_measure)

        # Находим лучшее k
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k_internal = k

        if has_true_labels and ari > best_ari:
            best_ari = ari
            best_k_external = k

    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print(f"   По внутренним метрикам: k={best_k_internal} (Silhouette: {best_silhouette:.3f})")
    if has_true_labels:
        print(f"   По внешним метрикам: k={best_k_external} (ARI: {best_ari:.3f})")

    # Строим графики
    if has_true_labels:
        _plot_minibatch_metrics(k_values, internal_metrics, external_metrics,
                                best_k_internal, best_k_external, texts, true_labels)  # Fixed: best_k_ext -> best_k_external
    else:
        _plot_minibatch_internal_metrics(k_values, internal_metrics, best_k_internal)

    if has_true_labels:
        return best_k_internal, best_k_external
    else:
        return best_k_internal


def _plot_minibatch_metrics(k_values, internal_metrics, external_metrics, best_k_int, best_k_ext, texts, true_labels):
    """
    Построение графиков для MiniBatchKMeans со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # График 1: Внутренние метрики
    ax1.plot(k_values, internal_metrics['silhouette'], 'bo-', label='Silhouette', linewidth=2)
    ax1.plot(k_values, _normalize(internal_metrics['calinski']), 'go-', label='Calinski (норм.)', linewidth=2)
    ax1.plot(k_values, _normalize([1 / d for d in internal_metrics['davies']]), 'ro-',
             label='1/Davies (норм.)', linewidth=2)
    ax1.axvline(x=best_k_int, color='blue', linestyle='--', alpha=0.7,
                label=f'Лучшее k={best_k_int} (внутр.)')
    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Нормализованные значения')
    ax1.set_title('MINIBATCHKMEANS: ВНУТРЕННИЕ МЕТРИКИ\n(↑ лучше)')
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
    ax2.set_title('MINIBATCHKMEANS: ВНЕШНИЕ МЕТРИКИ\n(↑ лучше)')
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
    ax3.set_title('MINIBATCHKMEANS: Silhouette vs ARI')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = []
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    for k in k_values:
        mbkmeans = MiniBatchKMeans(n_clusters=k, batch_size=100, random_state=42, n_init=3)
        start_time = time.time()
        mbkmeans.fit(X)
        times.append(time.time() - start_time)

    # ИСПРАВЛЕНИЕ: правильный формат цвета
    ax4.plot(k_values, times, color='purple', linestyle='-', marker='o', linewidth=2)
    ax4.set_xlabel('Количество кластеров (k)')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('MINIBATCHKMEANS: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_minibatch_internal_metrics(k_values, internal_metrics, best_k):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    # График 1: Все внутренние метрики
    ax1.plot(k_values, internal_metrics['silhouette'], 'bo-', label='Silhouette', linewidth=2)
    ax1.plot(k_values, _normalize(internal_metrics['calinski']), 'go-', label='Calinski (норм.)', linewidth=2)
    ax1.plot(k_values, _normalize([1 / d for d in internal_metrics['davies']]), 'ro-',
             label='1/Davies (норм.)', linewidth=2)
    ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                label=f'Лучшее k={best_k}')
    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Нормализованные значения')
    ax1.set_title('MINIBATCHKMEANS: ВНУТРЕННИЕ МЕТРИКИ\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Только silhouette для детального анализа
    ax2.plot(k_values, internal_metrics['silhouette'], 'bo-', linewidth=2, markersize=8)
    ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                label=f'Лучшее k={best_k}')
    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('MINIBATCHKMEANS: SILHOUETTE SCORE\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def _normalize(values):
    """Нормализация значений к диапазону [0, 1]"""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


# Простая функция для быстрой кластеризации MiniBatchKMeans
def simple_minibatch_cluster(texts, n_clusters=3, batch_size=100, max_iter=100):
    """
    Минимальная кластеризация текстов с MiniBatchKMeans
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    # MiniBatchKMeans кластеризация
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=42,
        max_iter=max_iter,
        n_init=3
    )
    labels = mbkmeans.fit_predict(X)

    # Вычисление метрик
    X_dense = X.toarray()

    metrics = {
        'silhouette': silhouette_score(X_dense, labels),
        'calinski_harabasz': calinski_harabasz_score(X_dense, labels),
        'davies_bouldin': davies_bouldin_score(X_dense, labels),
        'inertia': mbkmeans.inertia_
    }

    # Простой вывод
    print(f"📊 MiniBatchKMeans кластеризация {len(texts)} текстов на {n_clusters} кластеров:")
    print(f"🎯 МЕТРИКИ:")
    print(f"   Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.3f}")
    print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
    print(f"   Inertia: {metrics['inertia']:.1f}")

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

    # Информация о кластерах
    print(f"\n🔍 ИНФОРМАЦИЯ О КЛАСТЕРАХ:")
    for i in range(n_clusters):
        cluster_texts = [texts[j] for j, label in enumerate(labels) if label == i]
        print(f"🔸 Кластер {i}: {len(cluster_texts)} текстов")
        for text in cluster_texts[:2]:
            print(f"   - {text[:60]}..." if len(text) > 60 else f"   - {text}")
        if len(cluster_texts) > 2:
            print(f"   ... и еще {len(cluster_texts) - 2} текстов")
        print()

    return labels, metrics


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 MINIBATCHKMEANS ДЛЯ ТЕКСТОВ")
    print("=" * 50)

    # Вариант 1: Сравнение с внешними метриками
    print("🎯 ВАРИАНТ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_k_int, best_k_ext = compare_minibatch_kmeans_sizes(
        texts, true_labels, max_k=5, batch_size=50
    )

    # Вариант 2: Быстрая кластеризация
    print("\n🎯 ВАРИАНТ 2: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ")
    labels, metrics = simple_minibatch_cluster(texts, n_clusters=best_k_int, batch_size=50)