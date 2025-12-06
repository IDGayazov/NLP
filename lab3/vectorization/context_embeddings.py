import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Any, Union
from scipy.spatial.distance import cosine


class ContextualEmbeddings:
    """
    Класс для получения контекстных эмбеддингов с обработкой CUDA ошибок
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2", force_cpu: bool = True):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.force_cpu = force_cpu

        # Определяем устройство с обработкой ошибок
        self.device = self._get_device()

        self.load_model(model_name)

    def _get_device(self):
        """Безопасное определение устройства"""
        if self.force_cpu:
            print("🔧 Принудительное использование CPU")
            return torch.device("cpu")

        try:
            if torch.cuda.is_available():
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ CUDA доступна и работает")
                return torch.device("cuda")
            else:
                print("⚠️ CUDA недоступна, используем CPU")
                return torch.device("cpu")

        except Exception as e:
            print(f"⚠️ Ошибка CUDA: {e}, используем CPU")
            return torch.device("cpu")

    def load_model(self, model_name: str) -> None:
        """
        Загрузка модели с обработкой ошибок
        """
        print(f"🔄 Загрузка модели {model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Загружаем модель с указанием устройства
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Модель {model_name} загружена на {self.device}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            try:
                print("🔄 Пробуем загрузить на CPU...")
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(torch.device("cpu"))
                self.model.eval()
                self.device = torch.device("cpu")
                print("✅ Модель загружена на CPU")
            except Exception as e2:
                print(f"❌ Критическая ошибка загрузки: {e2}")

    def get_embeddings(self, texts: Union[str, List[str]],
                       pooling: str = "mean",
                       layers: Union[int, List[int]] = -1,
                       max_length: int = 512) -> Dict[str, Any]:
        """
        Получение эмбеддингов для текстов
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не загружена!")

        if isinstance(texts, str):
            texts = [texts]

        try:
            # Токенизация
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Перенос на устройство
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)

            # Извлечение эмбеддингов
            hidden_states = outputs.hidden_states

            # Определяем доступные слои
            available_layers = len(hidden_states)
            print(f"📊 Доступно слоев: {available_layers}")

            if isinstance(layers, int):
                layers = [layers]

            embeddings = {}
            for layer in layers:
                # Преобразуем отрицательные индексы в положительные
                if layer < 0:
                    layer_idx = available_layers + layer
                else:
                    layer_idx = layer

                # Проверяем, что слой существует
                if layer_idx < 0 or layer_idx >= available_layers:
                    print(f"⚠️ Слой {layer} недоступен. Используем последний слой.")
                    layer_idx = available_layers - 1

                layer_output = hidden_states[layer_idx]

                # Применяем пулинг
                if pooling == "mean":
                    attention_mask = encoded['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_output.size()).float()
                    sum_embeddings = torch.sum(layer_output * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    layer_embeddings = sum_embeddings / sum_mask

                elif pooling == "max":
                    attention_mask = encoded['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_output.size()).float()
                    layer_output[input_mask_expanded == 0] = -1e9
                    layer_embeddings = torch.max(layer_output, 1)[0]

                elif pooling == "cls":
                    layer_embeddings = layer_output[:, 0, :]

                elif pooling == "pooler" and hasattr(outputs, 'pooler_output'):
                    layer_embeddings = outputs.pooler_output
                else:
                    raise ValueError(f"Неизвестный метод пулинга: {pooling}")

                # Используем понятные ключи
                embeddings[f"layer_{layer_idx}"] = layer_embeddings.cpu().numpy()
                embeddings["last_layer"] = layer_embeddings.cpu().numpy()  # Псевдоним для последнего слоя

            return {
                'embeddings': embeddings,
                'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded['input_ids']],
                'pooling': pooling,
                'layers': layers,
                'available_layers': available_layers
            }

        except RuntimeError as e:
            if "CUDA" in str(e):
                print("⚠️ Ошибка CUDA, пробуем на CPU...")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                return self.get_embeddings(texts, pooling, layers, max_length)
            else:
                raise e


class SafeRussianBERTEmbeddings(ContextualEmbeddings):
    """Безопасная версия для русскоязычных моделей"""

    def __init__(self, model_size: str = "tiny"):
        """
        Args:
            model_size: 'tiny', 'base', 'large'
        """
        models = {
            "tiny": "cointegrated/rubert-tiny2",
            "base": "DeepPavlov/rubert-base-cased",
            "large": "sberbank-ai/ruBert-large"
        }

        if model_size not in models:
            raise ValueError(f"Доступные размеры: {list(models.keys())}")

        # Всегда используем CPU для безопасности
        super().__init__(models[model_size], force_cpu=True)

    def get_sentence_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести двух предложений"""
        emb1 = self.get_embeddings(text1, pooling="mean")
        emb2 = self.get_embeddings(text2, pooling="mean")

        # Используем последний слой (более безопасный доступ)
        vec1 = list(emb1['embeddings'].values())[0][0]  # Берем первый доступный слой
        vec2 = list(emb2['embeddings'].values())[0][0]

        # Косинусная схожесть
        similarity = 1 - cosine(vec1, vec2)
        return similarity


def demo_safe_embeddings():
    """
    Демонстрация работы на CPU
    """
    print("🎯 ДЕМОНСТРАЦИЯ КОНТЕКСТНЫХ ЭМБЕДДИНГОВ НА CPU")
    print("=" * 60)

    # Используем легкую модель на CPU
    embedder = SafeRussianBERTEmbeddings("tiny")

    # Тестовые тексты
    texts = [
        "Машинное обучение — это искусственный интеллект",
        "Нейронные сети используются для распознавания образов",
        "Сегодня хорошая погода в Казани",
        "Татарский язык очень красивый"
    ]

    print("\n1. 📊 ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ:")
    for text in texts:
        result = embedder.get_embeddings(text, pooling="mean")

        # Безопасное извлечение вектора
        emb_key = list(result['embeddings'].keys())[0]
        emb_vector = result['embeddings'][emb_key][0]

        print(f"   '{text[:40]}...'")
        print(f"      Ключ: {emb_key}, Размер: {emb_vector.shape}, Норма: {np.linalg.norm(emb_vector):.3f}")

    print("\n2. 🔍 СРАВНЕНИЕ ТЕКСТОВ:")
    text_pairs = [
        ("Машинное обучение", "Искусственный интеллект"),
        ("Машинное обучение", "Хорошая погода"),
        ("Казань", "Татарский язык")
    ]

    for text1, text2 in text_pairs:
        similarity = embedder.get_sentence_similarity(text1, text2)
        print(f"   '{text1}' vs '{text2}': {similarity:.3f}")

    print("\n3. 🎯 ИНФОРМАЦИЯ О МОДЕЛИ:")
    result = embedder.get_embeddings("тест", pooling="mean")
    print(f"   Доступные слои: {result['available_layers']}")
    print(f"   Доступные ключи эмбеддингов: {list(result['embeddings'].keys())}")

class NewsEmbeddingAnalyzer:
    """
    Анализатор эмбеддингов для новостей (безопасная версия)
    """

    def __init__(self, model_size: str = "tiny"):
        self.embedder = SafeRussianBERTEmbeddings(model_size)

    def analyze_news_batch(self, news_texts: List[str]) -> Dict[str, Any]:
        """
        Анализ батча новостей
        """
        print(f"📊 Анализ {len(news_texts)} новостей...")

        # Получаем эмбеддинги для всех новостей
        embeddings_result = self.embedder.get_embeddings(news_texts, pooling="mean")

        # Безопасное извлечение векторов
        emb_key = list(embeddings_result['embeddings'].keys())[0]
        news_vectors = embeddings_result['embeddings'][emb_key]

        # Анализ схожести
        analysis = self._analyze_similarity(news_texts, news_vectors)
        analysis['embeddings'] = news_vectors

        return analysis

    def _analyze_similarity(self, texts: List[str], vectors: np.ndarray) -> Dict[str, Any]:
        """Анализ попарной схожести текстов"""
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))

        # Вычисляем попарные схожести
        for i in range(n_texts):
            for j in range(n_texts):
                if i != j:
                    similarity_matrix[i][j] = 1 - cosine(vectors[i], vectors[j])

        # Находим наиболее похожие пары
        similar_pairs = []
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = similarity_matrix[i][j]
                similar_pairs.append({
                    'text1_index': i,
                    'text2_index': j,
                    'text1_preview': texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                    'text2_preview': texts[j][:50] + "..." if len(texts[j]) > 50 else texts[j],
                    'similarity': similarity
                })

        # Сортируем по убыванию схожести
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'similarity_matrix': similarity_matrix,
            'top_similar_pairs': similar_pairs[:5],
            'avg_similarity': np.mean(similarity_matrix[np.triu_indices(n_texts, k=1)]),
            'min_similarity': np.min(similarity_matrix[np.triu_indices(n_texts, k=1)]),
            'max_similarity': np.max(similarity_matrix[np.triu_indices(n_texts, k=1)])
        }

    def find_similar_news(self, target_text: str, candidate_texts: List[str], top_k: int = 3) -> List[tuple]:
        """
        Поиск наиболее похожих новостей на целевую
        """
        # Получаем эмбеддинг целевого текста
        target_emb = self.embedder.get_embeddings(target_text, pooling="mean")
        target_key = list(target_emb['embeddings'].keys())[0]
        target_vector = target_emb['embeddings'][target_key][0]

        # Получаем эмбеддинги кандидатов
        candidates_emb = self.embedder.get_embeddings(candidate_texts, pooling="mean")
        candidate_key = list(candidates_emb['embeddings'].keys())[0]
        candidate_vectors = candidates_emb['embeddings'][candidate_key]

        # Вычисляем схожести
        similarities = []
        for i, cand_vector in enumerate(candidate_vectors):
            similarity = 1 - cosine(target_vector, cand_vector)
            similarities.append((candidate_texts[i], similarity))

        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


def demo_with_sample_news():
    """
    Демонстрация с примером новостей
    """
    # Пример новостей
    sample_news = [
        "В Казани открылся новый технологический парк для IT-компаний",
        "Татарстан инвестирует в развитие искусственного интеллекта",
        "Погода в Казани: ожидается снег и похолодание",
        "Ученые разработали новую модель машинного обучения для анализа текстов",
        "В Татарстане проходит фестиваль татарской культуры и языка"
    ]

    analyzer = NewsEmbeddingAnalyzer("tiny")

    print("📰 АНАЛИЗ НОВОСТЕЙ С ПОМОЩЬЮ BERT")
    print("=" * 50)

    # Анализ всего набора новостей
    analysis = analyzer.analyze_news_batch(sample_news)

    print(f"\n📊 СТАТИСТИКА СХОЖЕСТИ:")
    print(f"   Средняя схожесть: {analysis['avg_similarity']:.3f}")
    print(f"   Минимальная схожесть: {analysis['min_similarity']:.3f}")
    print(f"   Максимальная схожесть: {analysis['max_similarity']:.3f}")

    print(f"\n🔍 САМЫЕ ПОХОЖИЕ НОВОСТИ:")
    for i, pair in enumerate(analysis['top_similar_pairs'], 1):
        print(f"   {i}. Схожесть: {pair['similarity']:.3f}")
        print(f"      📝 {pair['text1_preview']}")
        print(f"      📝 {pair['text2_preview']}")


if __name__ == "__main__":
    # Демонстрация на CPU
    demo_safe_embeddings()

    # Демонстрация с новостями
    demo_with_sample_news()

    # Интерактивный режим (раскомментируйте для использования)
    # interactive_safe_demo()