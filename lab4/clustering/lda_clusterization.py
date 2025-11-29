from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score)
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

from util.decribe import get_labels, get_texts


def preprocess_for_lda(texts):
    """
    Предобработка текстов для LDA
    """
    # Простая токенизация (можно добавать более сложную предобработку)
    tokenized_texts = [text.lower().split() for text in texts]
    return tokenized_texts


def compare_lda_models(texts, true_labels=None, num_topics_range=None, passes_range=None):
    """
    Сравнение LDA моделей с разным количеством тем и passes
    """
    # Предобработка текстов для LDA
    tokenized_texts = preprocess_for_lda(texts)

    # Создание словаря и корпуса
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)  # Фильтрация редких и частых слов
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

    has_true_labels = true_labels is not None

    if num_topics_range is None:
        num_topics_range = [2, 3, 5, 8, 10]
    if passes_range is None:
        passes_range = [5, 10]

    if has_true_labels:
        print("🔬 LDA MODEL С МЕТРИКАМИ:")
        print("Topics\tPasses\tCoherence\tPerplexity\tSilhouette\tARI\t\tNMI\t\tV-measure")
        print("-" * 95)
    else:
        print("🔬 LDA MODEL С МЕТРИКАМИ:")
        print("Topics\tPasses\tCoherence\tPerplexity\tSilhouette")
        print("-" * 65)

    results = []

    for num_topics in num_topics_range:
        for passes in passes_range:
            try:
                # Обучение LDA модели
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=passes,
                    random_state=42,
                    alpha='auto',
                    eta='auto'
                )

                # Вычисление метрик LDA
                # Когерентность тем
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()

                # Перплексия
                perplexity = lda_model.log_perplexity(corpus)

                # Получение распределений тем для документов
                topic_distributions = []
                for doc in corpus:
                    topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
                    topic_distributions.append([prob for _, prob in topic_dist])

                topic_distributions = np.array(topic_distributions)

                # Жесткое назначение тем (тема с максимальной вероятностью)
                hard_labels = np.argmax(topic_distributions, axis=1)

                # Внутренние метрики на основе распределений тем
                # Для silhouette нужны плотные векторы - используем распределения тем
                silhouette = silhouette_score(topic_distributions, hard_labels)

                if has_true_labels:
                    # Внешние метрики
                    ari = adjusted_rand_score(true_labels, hard_labels)
                    nmi = normalized_mutual_info_score(true_labels, hard_labels)
                    v_measure = v_measure_score(true_labels, hard_labels)

                    print(f"{num_topics}\t{passes}\t{coherence:.3f}\t\t{perplexity:.3f}\t\t{silhouette:.3f}\t\t"
                          f"{ari:.3f}\t\t{nmi:.3f}\t\t{v_measure:.3f}")
                else:
                    print(f"{num_topics}\t{passes}\t{coherence:.3f}\t\t{perplexity:.3f}\t\t{silhouette:.3f}")

                results.append({
                    'num_topics': num_topics,
                    'passes': passes,
                    'coherence': coherence,
                    'perplexity': perplexity,
                    'silhouette': silhouette,
                    'ari': ari if has_true_labels else -1,
                    'nmi': nmi if has_true_labels else -1,
                    'v_measure': v_measure if has_true_labels else -1,
                    'hard_labels': hard_labels,
                    'topic_distributions': topic_distributions,
                    'lda_model': lda_model,
                    'dictionary': dictionary,
                    'corpus': corpus
                })

            except Exception as e:
                print(f"{num_topics}\t{passes}\tERROR: {str(e)[:30]}...")

    # Находим лучшие параметры
    if results:
        best_by_coherence = max(results, key=lambda x: x['coherence'])
        best_by_silhouette = max(results, key=lambda x: x['silhouette'])

        if has_true_labels:
            best_by_ari = max(results, key=lambda x: x['ari'])

        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print(f"   По когерентности тем: {best_by_coherence['num_topics']} тем, "
              f"{best_by_coherence['passes']} passes "
              f"(Coherence: {best_by_coherence['coherence']:.3f})")
        print(f"   По внутренним метрикам: {best_by_silhouette['num_topics']} тем, "
              f"{best_by_silhouette['passes']} passes "
              f"(Silhouette: {best_by_silhouette['silhouette']:.3f})")
        if has_true_labels:
            print(f"   По внешним метрикам: {best_by_ari['num_topics']} тем, "
                  f"{best_by_ari['passes']} passes "
                  f"(ARI: {best_by_ari['ari']:.3f})")
    else:
        print(f"\n⚠️  Не удалось обучить LDA модели")
        best_by_coherence = best_by_silhouette = best_by_ari = None

    # Строим графики
    if has_true_labels and results:
        _plot_lda_metrics(results, texts, true_labels, best_by_coherence, best_by_ari, best_by_silhouette)
    elif results:
        _plot_lda_internal_metrics(results, best_by_coherence, best_by_silhouette)

    if has_true_labels and results:
        return best_by_coherence, best_by_ari, best_by_silhouette
    elif results:
        return best_by_coherence, best_by_silhouette
    else:
        return None, None, None


def _plot_lda_metrics(results, texts, true_labels, best_coherence, best_ari, best_silhouette):
    """
    Построение графиков для LDA со всеми метриками
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Подготовка данных для графиков
    passes_values = sorted(set(r['passes'] for r in results))
    topics_values = sorted(set(r['num_topics'] for r in results))

    # График 1: Coherence для разных passes
    for passes in passes_values:
        pass_data = [r for r in results if r['passes'] == passes]
        if pass_data:
            topics_vals = [r['num_topics'] for r in pass_data]
            coherence_vals = [r['coherence'] for r in pass_data]
            ax1.plot(topics_vals, coherence_vals, 'o-', linewidth=2, markersize=6,
                     label=f'passes={passes}')

    if best_coherence:
        ax1.axvline(x=best_coherence['num_topics'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее: {best_coherence["num_topics"]} тем')

    ax1.set_xlabel('Количество тем')
    ax1.set_ylabel('Coherence Score')
    ax1.set_title('LDA: COHERENCE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Perplexity
    for passes in passes_values:
        pass_data = [r for r in results if r['passes'] == passes]
        if pass_data:
            topics_vals = [r['num_topics'] for r in pass_data]
            perplexity_vals = [r['perplexity'] for r in pass_data]
            ax2.plot(topics_vals, perplexity_vals, 'o-', linewidth=2, markersize=6,
                     label=f'passes={passes}')

    ax2.set_xlabel('Количество тем')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('LDA: PERPLEXITY\n(↓ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Внешние метрики (ARI)
    if best_ari:
        for passes in passes_values:
            pass_data = [r for r in results if r['passes'] == passes and r['ari'] > -1]
            if pass_data:
                topics_vals = [r['num_topics'] for r in pass_data]
                ari_vals = [r['ari'] for r in pass_data]
                ax3.plot(topics_vals, ari_vals, 'o-', linewidth=2, markersize=6,
                         label=f'passes={passes}')

    ax3.set_xlabel('Количество тем')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('LDA: ADJUSTED RAND INDEX\n(↑ лучше)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # График 4: Время выполнения
    times = {}
    tokenized_texts = preprocess_for_lda(texts)
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

    for passes in passes_values:
        pass_times = []
        for num_topics in topics_values:
            try:
                start_time = time.time()
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=passes,
                    random_state=42
                )
                pass_times.append(time.time() - start_time)
            except:
                pass_times.append(np.nan)

        times[passes] = pass_times

    for passes, time_vals in times.items():
        ax4.plot(topics_values, time_vals, 'o-', linewidth=2, markersize=6, label=f'passes={passes}')

    ax4.set_xlabel('Количество тем')
    ax4.set_ylabel('Время выполнения (секунды)')
    ax4.set_title('LDA: ВРЕМЯ ВЫПОЛНЕНИЯ')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def _plot_lda_internal_metrics(results, best_coherence, best_silhouette):
    """
    Построение графиков только для внутренних метрик LDA
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    passes_values = sorted(set(r['passes'] for r in results))
    topics_values = sorted(set(r['num_topics'] for r in results))

    # График 1: Coherence Score
    for passes in passes_values:
        pass_data = [r for r in results if r['passes'] == passes]
        if pass_data:
            topics_vals = [r['num_topics'] for r in pass_data]
            coherence_vals = [r['coherence'] for r in pass_data]
            ax1.plot(topics_vals, coherence_vals, 'o-', linewidth=2, markersize=6,
                     label=f'passes={passes}')

    if best_coherence:
        ax1.axvline(x=best_coherence['num_topics'], color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее: {best_coherence["num_topics"]} тем')

    ax1.set_xlabel('Количество тем')
    ax1.set_ylabel('Coherence Score')
    ax1.set_title('LDA: COHERENCE SCORE\n(↑ лучше)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Silhouette Score
    for passes in passes_values:
        pass_data = [r for r in results if r['passes'] == passes]
        if pass_data:
            topics_vals = [r['num_topics'] for r in pass_data]
            silhouette_vals = [r['silhouette'] for r in pass_data]
            ax2.plot(topics_vals, silhouette_vals, 'o-', linewidth=2, markersize=6,
                     label=f'passes={passes}')

    if best_silhouette:
        ax2.axvline(x=best_silhouette['num_topics'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Лучшее: {best_silhouette["num_topics"]} тем')

    ax2.set_xlabel('Количество тем')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('LDA: SILHOUETTE SCORE\n(↑ лучше)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Простая функция для быстрой тематического моделирования LDA
def simple_lda_modeling(texts, num_topics=5, passes=10):
    """
    Минимальное тематическое моделирование с LDA
    """
    # Предобработка текстов
    tokenized_texts = preprocess_for_lda(texts)

    # Создание словаря и корпуса
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

    print(f"📊 LDA тематическое моделирование {len(texts)} текстов:")
    print(f"⚙️  ПАРАМЕТРЫ: {num_topics} тем, {passes} passes")
    print(f"📝 СЛОВАРЬ: {len(dictionary)} уникальных слов")

    # Обучение LDA модели
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
        alpha='auto',
        eta='auto'
    )

    # Вычисление метрик
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()
    perplexity = lda_model.log_perplexity(corpus)

    # Распределения тем для документов
    topic_distributions = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_distributions.append([prob for _, prob in topic_dist])

    topic_distributions = np.array(topic_distributions)
    hard_labels = np.argmax(topic_distributions, axis=1)

    # Внутренние метрики
    silhouette = silhouette_score(topic_distributions, hard_labels)

    metrics = {
        'coherence': coherence,
        'perplexity': perplexity,
        'silhouette': silhouette,
        'num_topics': num_topics,
        'passes': passes
    }

    print(f"🎯 МЕТРИКИ МОДЕЛИ:")
    print(f"   Coherence Score: {coherence:.3f}")
    print(f"   Perplexity: {perplexity:.3f}")
    print(f"   Silhouette Score: {silhouette:.3f}")

    # Интерпретация Coherence Score
    if coherence > 0.6:
        interpretation = "Отличная когерентность тем"
    elif coherence > 0.5:
        interpretation = "Хорошая когерентность тем"
    elif coherence > 0.4:
        interpretation = "Умеренная когерентность тем"
    else:
        interpretation = "Низкая когерентность тем"
    print(f"   Интерпретация: {interpretation}")

    # Визуализация тем
    print(f"\n🔍 ТЕМЫ И ИХ КЛЮЧЕВЫЕ СЛОВА:")
    topics = lda_model.print_topics(num_words=8)
    for idx, topic in topics:
        print(f"🔸 Тема {idx}: {topic}")

    # Анализ распределения документов по темам
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДОКУМЕНТОВ ПО ТЕМАМ:")
    unique_labels, counts = np.unique(hard_labels, return_counts=True)
    for topic_id, count in zip(unique_labels, counts):
        percentage = (count / len(texts)) * 100
        print(f"   Тема {topic_id}: {count} документов ({percentage:.1f}%)")

    # Примеры документов для каждой темы
    print(f"\n📄 ПРИМЕРЫ ДОКУМЕНТОВ ДЛЯ КАЖДОЙ ТЕМЫ:")
    for topic_id in range(num_topics):
        topic_docs = [texts[i] for i, label in enumerate(hard_labels) if label == topic_id]
        print(f"\n🔸 Тема {topic_id} ({len(topic_docs)} документов):")
        if topic_docs:
            for doc in topic_docs[:2]:  # Показываем первые 2 документа
                print(f"   - {doc[:80]}..." if len(doc) > 80 else f"   - {doc}")
            if len(topic_docs) > 2:
                print(f"   ... и еще {len(topic_docs) - 2} документов")

    # Анализ уверенности тематического назначения
    max_probs = np.max(topic_distributions, axis=1)
    confidence_stats = {
        'high_confidence': np.sum(max_probs > 0.8) / len(max_probs) * 100,
        'medium_confidence': np.sum((max_probs > 0.6) & (max_probs <= 0.8)) / len(max_probs) * 100,
        'low_confidence': np.sum(max_probs <= 0.6) / len(max_probs) * 100
    }

    print(f"\n🎯 УВЕРЕННОСТЬ ТЕМАТИЧЕСКОГО НАЗНАЧЕНИЯ:")
    print(f"   Высокая уверенность (>0.8): {confidence_stats['high_confidence']:.1f}% документов")
    print(f"   Средняя уверенность (0.6-0.8): {confidence_stats['medium_confidence']:.1f}% документов")
    print(f"   Низкая уверенность (≤0.6): {confidence_stats['low_confidence']:.1f}% документов")

    return hard_labels, topic_distributions, metrics, lda_model, dictionary


# Функция для предсказания тем новых документов
def predict_lda_topics(new_texts, lda_model, dictionary):
    """
    Предсказание тем для новых документов
    """
    tokenized_texts = preprocess_for_lda(new_texts)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

    topic_distributions = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_distributions.append([prob for _, prob in topic_dist])

    hard_labels = np.argmax(topic_distributions, axis=1)

    return hard_labels, np.array(topic_distributions)


if __name__ == "__main__":
    texts = get_texts()
    true_labels = get_labels()

    print("🚀 LDA THEMATIC MODELING ДЛЯ ТЕКСТОВ")
    print("=" * 60)

    # Вариант 1: Сравнение с метриками
    print("🎯 ВАРИАНТ 1: ПОЛНОЕ СРАВНЕНИЕ С МЕТРИКАМИ")
    best_by_coh, best_by_ext, best_by_sil = compare_lda_models(
        texts, true_labels,
        num_topics_range=[2, 3, 5, 8, 10],
        passes_range=[5, 10]
    )

    # Вариант 2: Быстрое тематическое моделирование
    print("\n🎯 ВАРИАНТ 2: БЫСТРОЕ ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ")
    if best_by_coh:
        hard_labels, topic_dists, metrics, lda_model, dictionary = simple_lda_modeling(
            texts,
            num_topics=best_by_coh['num_topics'],
            passes=best_by_coh['passes']
        )
    else:
        # Используем параметры по умолчанию если не нашли лучших
        hard_labels, topic_dists, metrics, lda_model, dictionary = simple_lda_modeling(
            texts,
            num_topics=5,
            passes=10
        )