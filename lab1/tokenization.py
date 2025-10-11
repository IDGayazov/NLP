import pandas as pd
import time
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Импорт библиотек для токенизации
import nltk
from nltk.tokenize import word_tokenize, regexp_tokenize
import spacy
import razdel

# Альтернативные стеммеры для тюркских языков
try:
    from snowballstemmer import stemmer
    SNOWBALL_AVAILABLE = True
except ImportError:
    print("⚠ SnowballStemmer не установлен. Установите: pip install snowballstemmer")
    SNOWBALL_AVAILABLE = False

# Для базовой морфологии можно попробовать pymorphy2 как fallback
try:
    import pymorphy2
    PYMOРHY_AVAILABLE = True
except ImportError:
    PYMOРHY_AVAILABLE = False

nltk.download('punkt')

class TatarTokenizationExperiment:
    def __init__(self, sample_texts):
        self.sample_texts = sample_texts
        self.results = []
        
        # Инициализация моделей для татарского
        self.available_methods = {
            'naive': True,
            'regex': True,
            'nltk': True,
            'spacy': False,
            'razdel': True,
            'snowball': False,
            'pymorphy': False,
            'rule_based': True  # Всегда доступен
        }
        
        self._init_tatar_models()
    
    def _init_tatar_models(self):
        """Инициализация моделей для татарского языка"""
        print("Инициализация моделей для татарского языка...")
        
        # SpaCy мультиязычная модель
        try:
            self.nlp_spacy = spacy.load("xx_ent_wiki_sm")
            self.available_methods['spacy'] = True
            print("✓ SpaCy мультиязычная модель загружена")
        except Exception as e:
            print(f"⚠ SpaCy модель не загружена: {e}")
        
        # SnowballStemmer для тюркских языков
        if SNOWBALL_AVAILABLE:
            try:
                # Пробуем турецкий стеммер (ближайший к татарскому)
                self.turkish_stemmer = stemmer('turkish')
                self.available_methods['snowball'] = True
                print("✓ SnowballStemmer (turkish) загружен")
            except Exception as e:
                print(f"⚠ Ошибка загрузки SnowballStemmer: {e}")
        
        # Pymorphy2 как fallback
        if PYMOРHY_AVAILABLE:
            try:
                self.morph = pymorphy2.MorphAnalyzer()
                self.available_methods['pymorphy'] = True
                print("✓ Pymorphy2 загружен (будет использован как базовый анализатор)")
            except Exception as e:
                print(f"⚠ Ошибка загрузки Pymorphy2: {e}")
        
        print("✓ Правиловый стеммер для татарского доступен")
    
    def preprocess_tatar_text(self, text):
        """Предобработка татарского текста"""
        # Сохраняем татарские буквы: ә, ө, ү, җ, ң, һ
        text = re.sub(r'[^\w\sәөүҗңһӘӨҮҖҢҺ]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def naive_tokenize(self, text):
        """Наивная токенизация по пробелам"""
        text = self.preprocess_tatar_text(text)
        return text.split()
    
    def regex_tokenize(self, text):
        """Токенизация на основе регулярных выражений для татарского"""
        text = self.preprocess_tatar_text(text)
        # Регулярное выражение для татарских слов (включая специфические буквы)
        return re.findall(r'[а-яәөүҗңһӘӨҮҖҢҺ]+', text)
    
    def nltk_tokenize(self, text):
        """Токенизация с помощью NLTK"""
        text = self.preprocess_tatar_text(text)
        return word_tokenize(text)
    
    def spacy_tokenize(self, text):
        """Токенизация с помощью SpaCy"""
        if not self.available_methods['spacy']:
            return self.regex_tokenize(text)
        
        doc = self.nlp_spacy(text)
        return [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    
    def razdel_tokenize(self, text):
        """Токенизация с помощью Razdel"""
        text = self.preprocess_tatar_text(text)
        return [token.text.lower() for token in razdel.tokenize(text)]
    
    def apply_turkish_stemming(self, tokens):
        """Стемминг с помощью турецкого стеммера"""
        if not self.available_methods['snowball']:
            return tokens
        
        try:
            stemmed_tokens = self.turkish_stemmer.stemWords(tokens)
            return stemmed_tokens
        except:
            return tokens
    
    def apply_rule_based_tatar_stemming(self, tokens):
        """Правиловый стеммер для татарского языка"""
        stemmed = []
        for token in tokens:
            stemmed_token = token
            
            # Удаляем множественное число
            if stemmed_token.endswith(('лар', 'ләр', 'нар', 'нәр')):
                stemmed_token = stemmed_token[:-3]
            # Удаляем падежные окончания
            elif stemmed_token.endswith(('ны', 'не', 'га', 'гә', 'ка', 'кә')):
                stemmed_token = stemmed_token[:-2]
            # Удаляем притяжательные аффиксы
            elif stemmed_token.endswith(('ым', 'ем', 'ың', 'ең')):
                stemmed_token = stemmed_token[:-2]
            # Удаляем другие частые аффиксы
            elif stemmed_token.endswith(('да', 'дә', 'та', 'тә')):
                stemmed_token = stemmed_token[:-2]
            elif stemmed_token.endswith(('дан', 'дән', 'тан', 'тән')):
                stemmed_token = stemmed_token[:-3]
            
            stemmed.append(stemmed_token)
        return stemmed
    
    def apply_pymorphy_lemmatization(self, tokens):
        """Лемматизация с помощью Pymorphy2 (как базовый вариант)"""
        if not self.available_methods['pymorphy']:
            return tokens
        
        lemmas = []
        for token in tokens:
            try:
                # Pymorphy2 для русского, но может работать с некоторыми татарскими словами
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                lemmas.append(lemma)
            except:
                lemmas.append(token)
        return lemmas
    
    def calculate_metrics(self, processed_texts, original_texts, method_name, processing_time):
        """Расчет метрик качества"""
        # Объём словаря
        all_tokens = []
        for text in processed_texts:
            all_tokens.extend(text)
        vocab_size = len(set(all_tokens))
        
        # Доля OOV
        reference_tokens = set()
        for text in original_texts:
            tokens = self.regex_tokenize(text)
            reference_tokens.update(tokens)
        
        unique_tokens = set(all_tokens)
        oov_tokens = unique_tokens - reference_tokens
        oov_rate = len(oov_tokens) / len(unique_tokens) if unique_tokens else 0
        
        # Степень сжатия словаря
        original_all_tokens = []
        for text in original_texts:
            original_all_tokens.extend(self.regex_tokenize(text))
        original_vocab_size = len(set(original_all_tokens))
        compression_rate = (original_vocab_size - vocab_size) / original_vocab_size if original_vocab_size > 0 else 0
        
        # Семантическая согласованность
        semantic_similarity = self.calculate_simple_similarity(original_texts, processed_texts)
        
        return {
            'method': method_name,
            'vocab_size': vocab_size,
            'oov_rate': round(oov_rate, 4),
            'compression_rate': round(compression_rate, 4),
            'processing_time_sec': round(processing_time, 4),
            'semantic_similarity': round(semantic_similarity, 4)
        }
    
    def calculate_simple_similarity(self, original_texts, processed_texts):
        """Упрощенный расчет семантического сходства"""
        try:
            similarities = []
            
            for i, (orig_text, proc_tokens) in enumerate(zip(original_texts, processed_texts)):
                orig_tokens = set(self.regex_tokenize(orig_text))
                proc_tokens_set = set(proc_tokens)
                
                if len(orig_tokens | proc_tokens_set) > 0:
                    jaccard_sim = len(orig_tokens & proc_tokens_set) / len(orig_tokens | proc_tokens_set)
                    similarities.append(jaccard_sim)
                else:
                    similarities.append(0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            print(f"Ошибка при расчете сходства: {e}")
            return 0.0
    
    def run_experiment(self):
        """Запуск полного эксперимента для татарского языка"""
        methods = [
            ('naive', 'none'),
            ('regex', 'none'),
            ('nltk', 'none'),
            ('spacy', 'none'),
            ('razdel', 'none'),
        ]
        
        # Добавляем методы с нормализацией
        methods.append(('razdel', 'rule_stemming'))  # Всегда доступен
        
        if self.available_methods['snowball']:
            methods.append(('razdel', 'turkish_stemming'))
        
        if self.available_methods['pymorphy']:
            methods.append(('razdel', 'pymorphy_lemmatization'))
        
        print(f"Будет протестировано {len(methods)} методов:")
        for method in methods:
            print(f"  - {method[0]} + {method[1]}")
        
        for tokenizer, normalizer in methods:
            print(f"Тестирование: {tokenizer} + {normalizer}")
            
            start_time = time.time()
            processed_texts = []
            
            for text in self.sample_texts:
                # Применяем токенизацию
                if tokenizer == 'naive':
                    tokens = self.naive_tokenize(text)
                elif tokenizer == 'regex':
                    tokens = self.regex_tokenize(text)
                elif tokenizer == 'nltk':
                    tokens = self.nltk_tokenize(text)
                elif tokenizer == 'spacy':
                    tokens = self.spacy_tokenize(text)
                elif tokenizer == 'razdel':
                    tokens = self.razdel_tokenize(text)
                
                # Применяем нормализацию
                if normalizer == 'rule_stemming':
                    tokens = self.apply_rule_based_tatar_stemming(tokens)
                elif normalizer == 'turkish_stemming':
                    tokens = self.apply_turkish_stemming(tokens)
                elif normalizer == 'pymorphy_lemmatization':
                    tokens = self.apply_pymorphy_lemmatization(tokens)
                
                processed_texts.append(tokens)
            
            processing_time = time.time() - start_time
            
            # Расчет метрик
            metrics = self.calculate_metrics(
                processed_texts, 
                self.sample_texts, 
                f"{tokenizer}_{normalizer}", 
                processing_time
            )
            
            self.results.append(metrics)
            print(f"  ✓ vocab={metrics['vocab_size']}, OOV={metrics['oov_rate']}, semantic={metrics['semantic_similarity']:.4f}")
        
        return pd.DataFrame(self.results)

# Тестовые данные на татарском языке
def load_tatar_sample_texts():
    """Загрузка примеров текстов на татарском языке"""
    sample_texts = [
        "Машина өйрәнү - ясалма интеллект бүлеге",
        "Нейрон челтәрләре образларны тану өчен кулланыла",
        "Тел эшкәртү компьютерларга кеше сөйләмен аңларга ярдәм итә",
        "Тирән өйрәнү зур исәпләү көче таләп итә",
        "Токенизация текстны эшкәртүнең мөһим этабы",
        "Лемматизация һәм стемминг сүзләрне нормальләштерүгә ярдәм итә",
        "Татар теле катлаулы морфологиягә ия",
        "Тәҗрибәләр төрле ысулларның эффективлыгын күрсәтә",
        "Сыйфат метрикалары иң яхшы подходны сайларга ярдәм итә",
        "Семантик охшашлык мәгънәнең саклануын үлчи"
    ]
    return sample_texts

def main():
    # Загрузка данных на татарском
    sample_texts = load_tatar_sample_texts()
    
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ: ТАРАР ТЕЛЕНДӘ ТОКЕНИЗАЦИЯ ҺӘМ НОРМАЛИЗАЦИЯ")
    print("=" * 60)
    
    # Запуск эксперимента
    experiment = TatarTokenizationExperiment(sample_texts)
    results_df = experiment.run_experiment()
    
    # Сохранение результатов
    results_df.to_csv('tatar_tokenization_metrics.csv', index=False, encoding='utf-8')
    
    # Вывод результатов
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ НӘТИҖӘЛӘРЕ")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Детальный анализ
    if len(results_df) > 0:
        print("\n" + "="*80)
        print("ДЕТАЛЬ АНАЛИЗ")
        print("="*80)
        
        # Находим лучшие методы по разным метрикам
        best_vocab = results_df.loc[results_df['vocab_size'].idxmin()]
        best_semantic = results_df.loc[results_df['semantic_similarity'].idxmax()]
        best_speed = results_df.loc[results_df['processing_time_sec'].idxmin()]
        best_compression = results_df.loc[results_df['compression_rate'].idxmax()]
        
        print(f"Иң кечкенә сүзлек: {best_vocab['method']} ({best_vocab['vocab_size']} токен)")
        print(f"Иң яхшы семантика: {best_semantic['method']} ({best_semantic['semantic_similarity']:.4f})")
        print(f"Иң тиз метод: {best_speed['method']} ({best_speed['processing_time_sec']:.4f} сек)")
        print(f"Иң яхшы сыгу: {best_compression['method']} ({best_compression['compression_rate']:.4f})")
        
        # Анализ методов с нормализацией
        normalized = results_df[~results_df['method'].str.contains('_none')]
        if len(normalized) > 0:
            print(f"\nНормализация методы (барлыгы {len(normalized)}):")
            for _, row in normalized.iterrows():
                print(f"  {row['method']:25} - OOV: {row['oov_rate']:.3f}, семантика: {row['semantic_similarity']:.3f}")

if __name__ == "__main__":
    main()