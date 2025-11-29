from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
import matplotlib.pyplot as plt

from util.decribe import get_texts, get_labels


def compare_cluster_sizes_with_external_metrics(texts, true_labels, max_k=6):
    """
    Сравнение кластеризации с внутренними и внешними метриками
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()

    print("🔬 СРАВНЕНИЕ РАЗЛИЧНЫХ K (ВНУТРЕННИЕ + ВНЕШНИЕ МЕТРИКИ):")
    print("k\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure")
    print("-" * 85)

    # Списки для хранения метрик
    k_values = []
    internal_metrics = {'silhouette': [], 'calinski': [], 'davies': []}
    external_metrics = {'ari': [], 'nmi': [], 'v_measure': []}

    best_k_internal = 2
    best_k_external = 2
    best_silhouette = -1
    best_ari = -1

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # Внутренние метрики (не требуют true_labels)
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        # Внешние метрики (требуют true_labels)
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

        # Находим лучшее k для внутренних и внешних метрик
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k_internal = k

        if ari > best_ari:
            best_ari = ari
            best_k_external = k

    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print(f"   По внутренним метрикам: k={best_k_internal} (Silhouette: {best_silhouette:.3f})")
    print(f"   По внешним метрикам: k={best_k_external} (ARI: {best_ari:.3f})")

    # Строим графики
    _plot_all_metrics(k_values, internal_metrics, external_metrics,
                      best_k_internal, best_k_external)

    return best_k_internal, best_k_external


def _plot_all_metrics(k_values, internal_metrics, external_metrics, best_k_int, best_k_ext):
    """
    Построение графиков для всех метрик
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


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 СРАВНЕНИЕ КЛАСТЕРОВ С ВНЕШНИМИ МЕТРИКАМИ")
    print("=" * 60)

    best_k_int, best_k_ext = compare_cluster_sizes_with_external_metrics(
        texts, true_labels, max_k=5
    )