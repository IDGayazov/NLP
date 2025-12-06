import torch
import numpy as np
from typing import List, Dict, Any, Union
from scipy.spatial.distance import cosine

from util.jsonl_process import read_jsonl_basic


class DocumentEmbeddingAggregator:
    """
    Класс для агрегации эмбеддингов токенов в вектор документа
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_name)

    def load_model(self, model_name: str):
        """Загрузка модели"""
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Модель {model_name} загружена на {self.device}")

    def get_document_embedding(self, text: str,
                               method: str = "mean",
                               layer: int = -1,
                               remove_special_tokens: bool = True) -> Dict[str, Any]:
        """
        Получение вектора документа из эмбеддингов токенов

        Args:
            text: входной текст
            method: метод агрегации ('mean', 'max', 'cls', 'pooler', 'weighted')
            layer: слой модели для извлечения
            remove_special_tokens: удалять ли специальные токены ([CLS], [SEP], [PAD])
        """
        # Токенизация
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Получение эмбеддингов
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)

        # Извлечение нужного слоя
        hidden_states = outputs.hidden_states
        layer_idx = layer if layer >= 0 else len(hidden_states) + layer
        token_embeddings = hidden_states[layer_idx]  # [batch_size, seq_len, hidden_size]

        # Применяем выбранный метод агрегации
        if method == "mean":
            doc_embedding = self._mean_pooling(token_embeddings, encoded['attention_mask'], remove_special_tokens)
        elif method == "max":
            doc_embedding = self._max_pooling(token_embeddings, encoded['attention_mask'], remove_special_tokens)
        elif method == "cls":
            doc_embedding = self._cls_pooling(token_embeddings)
        elif method == "pooler":
            doc_embedding = self._pooler_output(outputs)
        elif method == "weighted":
            doc_embedding = self._weighted_pooling(token_embeddings, encoded['attention_mask'], remove_special_tokens)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return {
            'document_embedding': doc_embedding.cpu().numpy(),
            'method': method,
            'layer': layer_idx,
            'token_embeddings': token_embeddings.cpu().numpy(),
            'tokens': self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        }

    def _mean_pooling(self, token_embeddings: torch.Tensor,
                      attention_mask: torch.Tensor,
                      remove_special_tokens: bool = True) -> torch.Tensor:
        """
        Усреднение эмбеддингов токенов
        """
        if remove_special_tokens:
            # Создаем маску для исключения специальных токенов
            input_mask = attention_mask.clone()
            special_tokens_mask = self._get_special_tokens_mask(attention_mask)
            input_mask[special_tokens_mask] = 0
        else:
            input_mask = attention_mask

        # Усреднение с учетом маски
        input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(self, token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor,
                     remove_special_tokens: bool = True) -> torch.Tensor:
        """
        Максимизация по каждому измерению
        """
        if remove_special_tokens:
            input_mask = attention_mask.clone()
            special_tokens_mask = self._get_special_tokens_mask(attention_mask)
            input_mask[special_tokens_mask] = 0
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Заменяем padding токены очень маленькими значениями
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    def _cls_pooling(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Использование [CLS] токена как представления документа
        """
        return token_embeddings[:, 0, :]

    def _pooler_output(self, outputs) -> torch.Tensor:
        """
        Использование pooler output (если доступен)
        """
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            raise ValueError("Pooler output не доступен для этой модели")

    def _weighted_pooling(self, token_embeddings: torch.Tensor,
                          attention_mask: torch.Tensor,
                          remove_special_tokens: bool = True) -> torch.Tensor:
        """
        Взвешенное усреднение с учетом IDF весов (упрощенная версия)
        """
        if remove_special_tokens:
            input_mask = attention_mask.clone()
            special_tokens_mask = self._get_special_tokens_mask(attention_mask)
            input_mask[special_tokens_mask] = 0
        else:
            input_mask = attention_mask

        # Простая эвристика: веса обратно пропорциональны частоте токена
        # В реальном сценарии нужно использовать предварительно вычисленные IDF веса
        weights = torch.ones_like(input_mask).float()

        # Уменьшаем вес для стоп-слов и знаков препинации
        tokens = self.tokenizer.convert_ids_to_tokens(attention_mask.nonzero()[:, 1])
        for i, token in enumerate(tokens):
            if token in ['.', ',', '!', '?', 'и', 'в', 'на', 'с']:
                weights[0, i] = 0.1

        input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        weights_expanded = weights.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded * weights_expanded, 1)
        sum_weights = torch.clamp((input_mask_expanded * weights_expanded).sum(1), min=1e-9)
        return sum_embeddings / sum_weights

    def _get_special_tokens_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Создание маски для специальных токенов
        """
        # Простая эвристика: первые и последние токены часто специальные
        batch_size, seq_len = attention_mask.shape
        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        # Помечаем первый токен ([CLS]) и padding токены
        special_mask[:, 0] = True  # [CLS]

        # Находим последний реальный токен ([SEP] или конец последовательности)
        for i in range(batch_size):
            real_tokens = attention_mask[i].nonzero()
            if len(real_tokens) > 0:
                last_token_idx = real_tokens[-1]
                special_mask[i, last_token_idx] = True  # [SEP] или последний токен

        return special_mask

    def compare_aggregation_methods(self, text: str) -> Dict[str, Any]:
        """
        Сравнение разных методов агрегации
        """
        methods = ["mean", "max", "cls", "weighted"]
        results = {}

        reference_method = "mean"
        reference_embedding = None

        for method in methods:
            try:
                result = self.get_document_embedding(text, method=method)
                embedding = result['document_embedding'][0]  # [hidden_size]
                results[method] = {
                    'embedding': embedding,
                    'shape': embedding.shape,
                    'norm': np.linalg.norm(embedding)
                }

                if method == reference_method:
                    reference_embedding = embedding

            except Exception as e:
                print(f"❌ Ошибка для метода {method}: {e}")
                results[method] = None

        # Вычисляем схожести с reference методом
        if reference_embedding is not None:
            for method, data in results.items():
                if data is not None and method != reference_method:
                    similarity = 1 - cosine(reference_embedding, data['embedding'])
                    data['similarity_with_mean'] = similarity

        return results


def demo_aggregation_methods():
    """
    Демонстрация разных методов агрегации
    """
    aggregator = DocumentEmbeddingAggregator()

    file_name1 = "../dataset/old_dataset.jsonl"
    file_name2 = "../dataset/new_dataset.jsonl"

    data1 = read_jsonl_basic(file_name1)
    data2 = read_jsonl_basic(file_name2)

    data = data1 + data2

    test_texts = []
    for item in data:
        test_texts.append(item['text'])

    print("🎯 ДЕМОНСТРАЦИЯ МЕТОДОВ АГРЕГАЦИИ ЭМБЕДДИНГОВ")
    print("=" * 60)

    # 1. Сравнение методов для одного текста
    print("\n1. 🔍 СРАВНЕНИЕ МЕТОДОВ ДЛЯ ОДНОГО ТЕКСТА:")
    sample_text = test_texts[0]
    methods_comparison = aggregator.compare_aggregation_methods(sample_text)

    for method, data in methods_comparison.items():
        if data:
            print(f"   {method:>10}: норма={data['norm']:.3f}", end="")
            if 'similarity_with_mean' in data:
                print(f", схожесть с mean={data['similarity_with_mean']:.3f}")
            else:
                print()

    # 2. Сравнение текстов с разными методами
    print("\n2. 📊 СХОЖЕСТЬ ТЕКСТОВ С РАЗНЫМИ МЕТОДАМИ:")
    methods = ["mean", "max", "cls"]

    for method in methods:
        print(f"\n   Метод: {method}")
        embeddings = []

        for text in test_texts:
            result = aggregator.get_document_embedding(text, method=method)
            embeddings.append(result['document_embedding'][0])

        # Вычисляем схожести
        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                print(f"      '{test_texts[i][:20]}...' vs '{test_texts[j][:20]}...': {similarity:.3f}")

    # 3. Визуализация токенов
    print("\n3. 🔤 АНАЛИЗ ТОКЕНОВ И ИХ ВКЛАДА:")
    analyze_token_contributions(aggregator, sample_text)


def analyze_token_contributions(aggregator: DocumentEmbeddingAggregator, text: str):
    """
    Анализ вклада отдельных токенов в вектор документа
    """
    print(f"   Текст: '{text}'")

    # Получаем эмбеддинги токенов
    result = aggregator.get_document_embedding(text, method="mean")
    token_embeddings = result['token_embeddings'][0]  # [seq_len, hidden_size]
    doc_embedding = result['document_embedding'][0]  # [hidden_size]
    tokens = result['tokens']

    print(f"   Токены: {tokens}")
    print(f"   Размерность токенов: {token_embeddings.shape}")
    print(f"   Размерность документа: {doc_embedding.shape}")

    # Вычисляем вклад каждого токена
    contributions = []
    for i, (token, token_emb) in enumerate(zip(tokens, token_embeddings)):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            # Схожесть токена с общим вектором документа
            similarity = 1 - cosine(token_emb, doc_embedding)
            contributions.append((token, similarity))

    # Сортируем по вкладу
    contributions.sort(key=lambda x: x[1], reverse=True)

    print(f"   Топ-5 самых важных токенов:")
    for token, contribution in contributions[:5]:
        print(f"      '{token}': {contribution:.3f}")


class AdvancedDocumentEmbeddings:
    """
    Продвинутые методы для работы с эмбеддингами документов
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.aggregator = DocumentEmbeddingAggregator(model_name)

    def get_batch_embeddings(self, texts: List[str],
                             method: str = "mean",
                             batch_size: int = 8) -> np.ndarray:
        """
        Получение эмбеддингов для батча текстов
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                result = self.aggregator.get_document_embedding(text, method=method)
                batch_embeddings.append(result['document_embedding'][0])

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def find_similar_documents(self, query: str,
                               documents: List[str],
                               method: str = "mean",
                               top_k: int = 5) -> List[tuple]:
        """
        Поиск наиболее похожих документов на запрос
        """
        # Получаем эмбеддинг запроса
        query_result = self.aggregator.get_document_embedding(query, method=method)
        query_embedding = query_result['document_embedding'][0]

        # Получаем эмбеддинги документов
        doc_embeddings = self.get_batch_embeddings(documents, method=method)

        # Вычисляем схожести
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = 1 - cosine(query_embedding, doc_emb)
            similarities.append((documents[i], similarity))

        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def document_clustering(self, texts: List[str],
                            method: str = "mean",
                            n_clusters: int = 3) -> Dict[str, Any]:
        """
        Кластеризация документов по их эмбеддингам
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Получаем эмбеддинги
        embeddings = self.get_batch_embeddings(texts, method=method)

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Оценка качества
        silhouette_avg = silhouette_score(embeddings, clusters)

        # Группируем документы по кластерам
        clustered_docs = {}
        for i, (text, cluster) in enumerate(zip(texts, clusters)):
            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(text)

        return {
            'clusters': clustered_docs,
            'embeddings': embeddings,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg
        }


# Пример использования с новостями
def demo_news_analysis():
    """
    Демонстрация анализа новостей с помощью эмбеддингов документов
    """
    # news_articles = [
    #     "В Казани открылся новый технологический парк для IT-компаний. Инвестиции составили 2 миллиарда рублей.",
    #     "Татарстан активно инвестирует в развитие искусственного интеллекта и машинного обучения.",
    #     "Погода в Казани: на этой неделе ожидается снег и похолодание до -15 градусов.",
    #     "Ученые Казанского университета разработали новую модель машинного обучения для анализа текстов.",
    #     "В Татарстане проходит ежегодный фестиваль татарской культуры и языка.",
    #     "Нейронные сети помогают врачам в диагностике заболеваний по медицинским снимкам.",
    #     "Казань становится центром IT-разработки в Поволжье, привлекая молодых специалистов."
    # ]

    file_name1 = "../dataset/old_dataset.jsonl"
    file_name2 = "../dataset/new_dataset.jsonl"

    data1 = read_jsonl_basic(file_name1)
    data2 = read_jsonl_basic(file_name2)

    data = data1 + data2

    news_articles = []
    for item in data:
        news_articles.append(item['text'])

    advanced_emb = AdvancedDocumentEmbeddings()

    print("📰 АНАЛИЗ НОВОСТЕЙ С ПОМОЩЬЮ ЭМБЕДДИНГОВ ДОКУМЕНТОВ")
    print("=" * 60)

    # 1. Поиск похожих новостей
    print("\n1. 🔍 ПОИСК ПОХОЖИХ НОВОСТЕЙ:")
    query = "технологии и искусственный интеллект"
    similar_news = advanced_emb.find_similar_documents(query, news_articles, top_k=3)

    print(f"   Запрос: '{query}'")
    for i, (news, similarity) in enumerate(similar_news, 1):
        print(f"   {i}. {similarity:.3f} - {news[:60]}...")

    # 2. Кластеризация новостей
    print("\n2. 🎯 КЛАСТЕРИЗАЦИЯ НОВОСТЕЙ:")
    clustering_result = advanced_emb.document_clustering(news_articles, n_clusters=3)

    print(f"   Качество кластеризации (silhouette): {clustering_result['silhouette_score']:.3f}")

    for cluster_id, docs in clustering_result['clusters'].items():
        print(f"\n   Кластер {cluster_id}:")
        for doc in docs[:2]:  # Покажем по 2 документа из каждого кластера
            print(f"      - {doc[:50]}...")


if __name__ == "__main__":
    # Демонстрация методов агрегации
    demo_aggregation_methods()

    # Демонстрация анализа новостей
    demo_news_analysis()