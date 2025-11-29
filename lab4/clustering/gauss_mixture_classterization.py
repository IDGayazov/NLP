from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time

from util.decribe import get_labels, get_texts


def compare_gaussian_mixture(texts, true_labels=None, max_k=6, covariance_types=None):
    """
    Сравнение GaussianMixture с разными типами ковариационных матриц
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    # Стандартизация для GaussianMixture (важно для сходимости)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    has_true_labels = true_labels is not None

    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']

    if has_true_labels:
        print("🔬 GAUSSIAN MIXTURE С ВНУТРЕННИМИ И ВНЕШНИМИ МЕТРИКАМИ:")
        print("k\tCovariance\tSilhouette\tCalinski\tDavies-B\tARI\t\tNMI\t\tV-measure\tConverged")
        print("-" * 115)
    else:
        print("🔬 GAUSSIAN MIXTURE С ВНУТРЕННИМИ МЕТРИКАМИ:")
        print("k\tCovariance\tSilhouette\tCalinski\tDavies-B\tConverged")
        print("-" * 85)

    results = []

    for k in range(2, max_k + 1):
        for covariance_type in covariance_types:
            try:
                # GaussianMixture кластеризация
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    random_state=42,
                    max_iter=100,
                    n_init=3
                )
                # Мягкое назначение - вероятности принадлежности
                soft_labels = gmm.fit_predict(X_scaled)
                # Жесткое назначение для метрик
                hard_labels = gmm.predict(X_scaled)

                # Проверяем сходимость
                converged = gmm.converged_
                n_iter = gmm.n_iter_

                # Внутренние метрики
                silhouette = silhouette_score(X_dense, hard_labels)
                calinski = calinski_harabasz_score(X_dense, hard_labels)
                davies = davies_bouldin_score(X_dense, hard_labels)

                if has_true_labels:
                    # Внешние метрики
                    ari = adjusted_rand_score(true_labels, hard_labels)
                    nmi = normalized_mutual_info_score(true_labels, hard_labels)
                    v_measure = v_measure_score(true_labels, hard_labels)

                    print(f"{k}\t{covariance_type}\t\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
                          f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}\t\t{converged}")
                else:
                    print(f"{k}\t{covariance_type}\t\t{silhouette:.3f}\t\t{calinski:.3f}\t\t{davies:.3f}\t\t"
                          f"{converged}")

                # Сохраняем вероятности принадлежности
                probabilities = gmm.predict_proba(X_scaled)

                results.append({
                    'k': k,
                    'covariance_type': covariance_type,
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies,
                    'ari': ari if has_true_labels else -1,
                    'nmi': nmi if has_true_labels else -1,
                    'v_measure': v_measure if has_true_labels else -1,
                    'hard_labels': hard_labels,
                    'soft_labels': soft_labels,
                    'probabilities': probabilities,
                    'converged': converged,
                    'n_iter': n_iter,
                    'bic': gmm.bic(X_scaled),
                    'aic': gmm.aic(X_scaled),
                    'gmm': gmm
                })

            except Exception as e:
                print(f"{k}\t{covariance_type}\t\tERROR: {str(e)[:30]}...")

    # Находим лучшие параметры
    if results:
        # Фильтруем только сходившиеся модели
        converged_results = [r for r in results if r['converged']]
        if converged_results:
            best_by_silhouette = max(converged_results, key=lambda x: x['silhouette'])
            best_by_bic = min(converged_results, key=lambda x: x['bic'])  # BIC - чем меньше, тем лучше

            if has_true_labels:
                best_by_ari = max(converged_results, key=lambda x: x['ari'])

            print(f"\n🎯 РЕКОМЕНДАЦИИ:")
            print(f"   По внутренним метрикам: k={best_by_silhouette['k']}, "
                  f"covariance={best_by_silhouette['covariance_type']} "
                  f"(Silhouette: {best_by_silhouette['silhouette']:.3f})")
            print(f"   По BIC: k={best_by_bic['k']}, covariance={best_by_bic['covariance_type']} "
                  f"(BIC: {best_by_bic['bic']:.1f})")
            if has_true_labels:
                print(f"   По внешним метрикам: k={best_by_ari['k']}, "
                      f"covariance={best_by_ari['covariance_type']} "
                      f"(ARI: {best_by_ari['ari']:.3f})")
        else:
            print(f"\n⚠️  Ни одна модель не сошлась")
            best_by_silhouette = best_by_bic = best_by_ari = None
    else:
        print(f"\n⚠️  Не удалось выполнить кластеризацию")
        best_by_silhouette = best_by_bic = best_by_ari = None

    # Строим графики
    if has_true_labels and results:
        _plot_gmm_metrics(results, texts, true_labels, best_by_silhouette, best_by_ari, best_by_bic)
    elif results:
        _plot_gmm_internal_metrics(results, best_by_silhouette, best_by_bic)

    if has_true_labels and converged_results:
        return best_by_silhouette, best_by_ari, best_by_bic
    elif converged_results:
        return best_by_silhouette, best_by_bic
    else:
        return None, None, None


def _plot_gmm_metrics(results, texts, true_labels, best_silhouette, best_ari, best_bic):
    """
    Построение графиков для GaussianMixture со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Подготовка данных для графиков
    covariance_types = sorted(set(r['covariance_type'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score для разных covariance types
    for cov_type in covariance_types:
        cov_data = [r for r in results if r['covariance_type'] == cov_type and r['converged']]
        if cov_data:
            k_vals = [r['k'] for r in cov_data]
            silhouette_vals = [r['silhouette'] for r in cov_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{cov_type}')

    if best_silhouette:
        ax1.axvline(x=best_silhouette['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_silhouette["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('GAUSSIAN MIXTURE: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: BIC (Bayesian Information Criterion)
    for cov_type in covariance_types:
        cov_data = [r for r in results if r['covariance_type'] == cov_type and r['converged']]
        if cov_data:
            k_vals = [r['k'] for r in cov_data]
            bic_vals = [r['bic'] for r in cov_data]
            ax2.plot(k_vals, bic_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{cov_type}')

    if best_bic:
        ax2.axvline(x=best_bic['k'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_bic["k"]} (BIC)')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('BIC Score')
    ax2.set_title('GAUSSIAN MIXTURE: BAYESIAN INFORMATION CRITERION\n(↓ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Внешние метрики (ARI)
    if best_ari:
        for cov_type in covariance_types:
            cov_data = [r for r in results if r['covariance_type'] == cov_type and r['converged'] and r['ari'] > -1]
            if cov_data:
                k_vals = [r['k'] for r in cov_data]
                ari_vals = [r['ari'] for r in cov_data]
                ax3.plot(k_vals, ari_vals, 'o-', linewidth=2, markersize=6,
                         label=f'{cov_type}')

    ax3.set_xlabel('Количество кластеров (k)')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('GAUSSIAN MIXTURE: ADJUSTED RAND INDEX\n(↑ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = {}
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    for cov_type in covariance_types:
        cov_times = []
        for k in k_values:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                      random_state=42, max_iter=50)
                start_time = time.time()
                gmm.fit(X_scaled)
                cov_times.append(time.time() - start_time)
            except:
                cov_times.append(np.nan)

        times[cov_type] = cov_times

    for cov_type, time_vals in times.items():
        ax4.plot(k_values, time_vals, 'o-', linewidth=2, markersize=6, label=cov_type)

    ax4.set_xlabel('Количество кластеров (k)')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('GAUSSIAN MIXTURE: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def _plot_gmm_internal_metrics(results, best_silhouette, best_bic):
    """
    Построение графиков только для внутренних метрик
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    covariance_types = sorted(set(r['covariance_type'] for r in results))
    k_values = sorted(set(r['k'] for r in results))

    # График 1: Silhouette Score
    for cov_type in covariance_types:
        cov_data = [r for r in results if r['covariance_type'] == cov_type and r['converged']]
        if cov_data:
            k_vals = [r['k'] for r in cov_data]
            silhouette_vals = [r['silhouette'] for r in cov_data]
            ax1.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'{cov_type}')

    if best_silhouette:
        ax1.axvline(x=best_silhouette['k'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_silhouette["k"]}')

    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('GAUSSIAN MIXTURE: SILHOUETTE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: BIC и AIC
    for cov_type in covariance_types:
        cov_data = [r for r in results if r['covariance_type'] == cov_type and r['converged']]
        if cov_data:
            k_vals = [r['k'] for r in cov_data]
            bic_vals = [r['bic'] for r in cov_data]
            aic_vals = [r['aic'] for r in cov_data]

            ax2.plot(k_vals, bic_vals, 'o-', linewidth=2, markersize=6,
                     label=f'BIC ({cov_type})')
            ax2.plot(k_vals, aic_vals, 'o--', linewidth=1, markersize=4,
                     label=f'AIC ({cov_type})', alpha=0.7)

    if best_bic:
        ax2.axvline(x=best_bic['k'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Лучшее k={best_bic["k"]}')

    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('BIC / AIC Score')
    ax2.set_title('GAUSSIAN MIXTURE: BIC И AIC\n(↓ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Простая функция для быстрой кластеризации GaussianMixture
def simple_gmm_cluster(texts, n_components=3, covariance_type='full'):
    """
    Минимальная кластеризация текстов с GaussianMixture (мягкое назначение)
    """
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    # Стандартизация для GaussianMixture
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

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

    # Вычисление метрик
    metrics = {
        'silhouette': silhouette_score(X_dense, hard_labels),
        'calinski_harabasz': calinski_harabasz_score(X_dense, hard_labels),
        'davies_bouldin': davies_bouldin_score(X_dense, hard_labels),
        'bic': gmm.bic(X_scaled),
        'aic': gmm.aic(X_scaled),
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
        'covariance_type': covariance_type
    }

    # Простой вывод
    print(f"📊 GaussianMixture кластеризация {len(texts)} текстов на {n_components} кластеров:")
    print(f"⚙️  ПАРАМЕТРЫ: covariance_type={covariance_type}")
    print(f"🎯 МЕТРИКИ:")
    print(f"   Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.3f}")
    print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
    print(f"   BIC: {metrics['bic']:.1f}")
    print(f"   AIC: {metrics['aic']:.1f}")
    print(f"   Сходимость: {metrics['converged']} (итераций: {metrics['n_iter']})")

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

    # Информация о типах ковариационных матриц
    print(f"\n📋 ТИПЫ КОВАРИАЦИОННЫХ МАТРИЦ:")
    cov_info = {
        'full': "Полная ковариационная матрица для каждого кластера",
        'tied': "Одна общая ковариационная матрица для всех кластеров",
        'diag': "Диагональная ковариационная матрица для каждого кластера",
        'spherical': "Сферическая ковариационная матрица (одинаковая по всем направлениям)"
    }
    print(f"   {covariance_type}: {cov_info.get(covariance_type, '')}")

    # Анализ мягкого назначения
    print(f"\n🔮 МЯГКОЕ НАЗНАЧЕНИЕ КЛАСТЕРОВ:")
    print(f"   Каждая точка имеет вероятности принадлежности ко всем кластерам")

    # Показываем примеры мягкого назначения для нескольких точек
    print(f"\n📊 ПРИМЕРЫ ВЕРОЯТНОСТЕЙ ПРИНАДЛЕЖНОСТИ (первые 5 точек):")
    print("Точка\t" + "\t".join([f"Кластер {i}" for i in range(n_components)]))
    for i in range(min(5, len(probabilities))):
        prob_str = "\t".join([f"{p:.3f}" for p in probabilities[i]])
        print(f"{i}\t{prob_str}")

    # Анализ уверенности классификации
    max_probs = np.max(probabilities, axis=1)
    confidence_stats = {
        'high_confidence': np.sum(max_probs > 0.9) / len(max_probs) * 100,
        'medium_confidence': np.sum((max_probs > 0.7) & (max_probs <= 0.9)) / len(max_probs) * 100,
        'low_confidence': np.sum(max_probs <= 0.7) / len(max_probs) * 100
    }

    print(f"\n🎯 УВЕРЕННОСТЬ КЛАССИФИКАЦИИ:")
    print(f"   Высокая уверенность (>0.9): {confidence_stats['high_confidence']:.1f}% точек")
    print(f"   Средняя уверенность (0.7-0.9): {confidence_stats['medium_confidence']:.1f}% точек")
    print(f"   Низкая уверенность (≤0.7): {confidence_stats['low_confidence']:.1f}% точек")

    # Информация о кластерах (жесткое назначение)
    print(f"\n🔍 ИНФОРМАЦИЯ О КЛАСТЕРАХ (жесткое назначение):")
    unique_labels = np.unique(hard_labels)
    for i in unique_labels:
        cluster_texts = [texts[j] for j, label in enumerate(hard_labels) if label == i]
        avg_confidence = np.mean(max_probs[hard_labels == i])

        print(f"🔸 Кластер {i}: {len(cluster_texts)} текстов (средняя уверенность: {avg_confidence:.3f})")
        if len(cluster_texts) > 0:
            for text in cluster_texts[:2]:
                print(f"   - {text[:60]}..." if len(text) > 60 else f"   - {text}")
            if len(cluster_texts) > 2:
                print(f"   ... и еще {len(cluster_texts) - 2} текстов")
        print()

    return hard_labels, soft_labels, probabilities, metrics, gmm


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 GAUSSIAN MIXTURE ДЛЯ ТЕКСТОВ")
    print("=" * 60)

    # Вариант 1: Сравнение с внешними метриками
    print("🎯 ВАРИАНТ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_int, best_by_ext, best_by_bic = compare_gaussian_mixture(
        texts, true_labels, max_k=5,
        covariance_types=['full', 'tied', 'diag', 'spherical']
    )

    # Вариант 2: Быстрая кластеризация
    print("\n🎯 ВАРИАНТ 2: БЫСТРАЯ КЛАСТЕРИЗАЦИЯ С МЯГКИМ НАЗНАЧЕНИЕМ")
    if best_by_int:
        hard_labels, soft_labels, probabilities, metrics, gmm = simple_gmm_cluster(
            texts,
            n_components=best_by_int['k'],
            covariance_type=best_by_int['covariance_type']
        )
    else:
        # Используем параметры по умолчанию если не нашли лучших
        hard_labels, soft_labels, probabilities, metrics, gmm = simple_gmm_cluster(
            texts,
            n_components=3,
            covariance_type='full'
        )