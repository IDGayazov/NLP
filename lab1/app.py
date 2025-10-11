import streamlit as st
import nltk
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import time
import os
import html
import json
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup

# Subword модели
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers
import sentencepiece as spm
from text_cleaner import TextCleaner
from universal_preprocessor import UniversalPreprocessor

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

# Sample datasets with Tatar language
SAMPLE_DATASETS = {
    "english_news": {
        "name": "English News Headlines",
        "data": [
            "Breaking news: Scientists discover new species in Amazon rainforest",
            "Stock markets reach all-time high amid economic recovery",
            "Climate change conference concludes with new agreements",
            "Technology company unveils revolutionary AI system",
            "Global health organization reports progress in disease prevention"
        ]
    },
    "russian_literature": {
        "name": "Russian Literature Excerpts", 
        "data": [
            "В тот день, когда я впервые увидел её, солнце светило особенно ярко.",
            "Он долго шёл по пустынной улице, размышляя о жизни и её смысле.",
            "Тишина в доме была звенящей, нарушаемой лишь тиканьем старых часов.",
            "Она открыла книгу и погрузилась в мир, созданный воображением автора.",
            "Ветер гнал по небу рваные облака, предвещая скорый дождь."
        ]
    },
    "tatar_texts": {
        "name": "Tatar Language Examples",
        "data": [
            "Татар теле - боек һәм бай тел, аның тарихы бик борынгы.",
            "Казан шәһәре Идел буенда урнашкан матур шәһәр.",
            "Татар халкының мәдәнияте һәм гореф-гадәтләре бик бай.",
            "Әдәбият безнең милләтебезнең күңелен ача.",
            "Тел - халыкның рухи байлыгы, аны сакларга кирәк."
        ]
    },
    "tech_articles": {
        "name": "Technology Articles",
        "data": [
            "Artificial intelligence is transforming modern healthcare with innovative solutions.",
            "Blockchain technology provides secure and transparent transaction systems.",
            "Cloud computing enables scalable and flexible infrastructure for businesses.",
            "Machine learning algorithms can predict consumer behavior with high accuracy.",
            "Cybersecurity measures are essential for protecting digital assets."
        ]
    }
}

class TextProcessor:
    def __init__(self):
        self.stopwords = {
            'en': set(nltk.corpus.stopwords.words('english')),
            'ru': set(nltk.corpus.stopwords.words('russian')),
            'tt': TextCleaner(language='tatar')._get_tatar_stopwords()
        }
    
    def tokenize(self, text, method='word', language='en'):
        """Tokenize text using different methods"""
        text = str(text)
        if method == 'word':
            if language == 'en':
                tokens = re.findall(r'\b\w+\b', text.lower())
            elif language == 'ru':
                tokens = re.findall(r'\b[а-яё]+\b', text.lower())
            elif language == 'tt':
                # Татарский язык: кириллица + специфические символы
                tokens = re.findall(r'\b[а-яәөүҗңһ]+\b', text.lower())
            else:
                tokens = re.findall(r'\b\w+\b', text.lower())
        elif method == 'nltk':
            tokens = nltk.word_tokenize(text.lower())
        else:
            tokens = text.split()
        
        return tokens
    
    def normalize(self, tokens, normalization='lowercase'):
        """Normalize tokens"""
        if not tokens:
            return []
            
        if normalization == 'lowercase':
            return [token.lower() for token in tokens]
        elif normalization == 'stemming':
            stemmer = nltk.stem.PorterStemmer()
            return [stemmer.stem(token) for token in tokens]
        elif normalization == 'lemmatization':
            lemmatizer = nltk.stem.WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in tokens]
        else:
            return tokens
    
    def remove_stopwords(self, tokens, language='en'):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stopwords.get(language, set())]

class SubwordModelComparator:
    def __init__(self, corpus: List[str]):
        # Фильтруем и очищаем корпус
        self.corpus = [text.strip() for text in corpus if text.strip()]
        self.results = []
        
    def prepare_corpus_file(self):
        """Сохраняет корпус во временный файл для sentencepiece"""
        if not self.corpus:
            raise ValueError("Корпус пуст после фильтрации")
            
        with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
            for text in self.corpus:
                f.write(text + '\n')
        return 'temp_corpus.txt'

    def calculate_fragmentation(self, tokenized_texts: List[List[str]]) -> float:
        """Вычисляет процент фрагментации слов"""
        total_words = 0
        fragmented_words = 0

        for tokens in tokenized_texts:
            for token in tokens:
                # Определяем фрагментированные токены
                if (token.startswith('##') or '▁' in token or 
                    (len(token) < 3 and token not in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]'])):
                    fragmented_words += 1
                total_words += 1

        return (fragmented_words / total_words * 100) if total_words > 0 else 0

    def calculate_compression_ratio(self, original_texts: List[str], tokenized_texts: List[List[str]]) -> float:
        """Вычисляет коэффициент сжатия"""
        total_original_tokens = sum(len(text.split()) for text in original_texts if text.strip())
        total_subword_tokens = sum(len(tokens) for tokens in tokenized_texts)

        return total_subword_tokens / total_original_tokens if total_original_tokens > 0 else 1

    def normalize_text(self, text: str) -> str:
        """Нормализует текст"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def reconstruct_text_for_model(self, tokens: List[str], model_name: str) -> str:
        """Правильная реконструкция текста для каждой модели"""
        if not tokens:
            return ""

        try:
            if model_name == "Unigram_SP":
                text = ''.join(tokens).replace('▁', ' ').strip()
            elif model_name == "WordPiece":
                if not tokens:
                    return ""
                text = tokens[0]
                for token in tokens[1:]:
                    if token.startswith('##'):
                        text += token[2:]
                    else:
                        text += ' ' + token
            elif model_name == "BPE":
                text = ' '.join(tokens).replace(' ##', '')
            elif model_name == "Unigram_HF":
                text = ' '.join(tokens).replace(' ##', '')
            else:
                text = ' '.join(tokens)
        except Exception as e:
            st.error(f"Ошибка реконструкции для {model_name}: {e}")
            text = ' '.join(tokens)

        return self.normalize_text(text)

    def calculate_reconstruction_accuracy(self, original_texts: List[str], reconstructed_texts: List[str]) -> float:
        """Вычисляет точность реконструкции"""
        correct = 0
        total = min(len(original_texts), len(reconstructed_texts))

        for i in range(total):
            orig = original_texts[i]
            rec = reconstructed_texts[i]

            if not orig.strip() or not rec.strip():
                continue

            orig_norm = self.normalize_text(orig)
            rec_norm = self.normalize_text(rec)

            if orig_norm == rec_norm:
                correct += 1

        accuracy = (correct / total * 100) if total > 0 else 0
        return accuracy

    def debug_tokenization(self, model_name: str, tokenized_texts: List[List[str]], num_examples: int = 1):
        """Показывает примеры токенизации для отладки"""
        with st.expander(f"🔍 Примеры токенизации ({model_name})"):
            for i in range(min(num_examples, len(tokenized_texts))):
                if i < len(self.corpus):
                    original = self.corpus[i]
                    tokens = tokenized_texts[i]
                    reconstructed = self.reconstruct_text_for_model(tokens, model_name)

                    st.write(f"**Пример {i+1}:**")
                    st.text(f"Оригинал: {original[:100]}{'...' if len(original) > 100 else ''}")
                    st.text(f"Токены: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
                    st.text(f"Восстановленный: {reconstructed[:100]}{'...' if len(reconstructed) > 100 else ''}")
                    st.text(f"Совпадение: {self.normalize_text(original) == self.normalize_text(reconstructed)}")
                    st.write("---")

    def train_bpe(self, vocab_size: int, min_frequency: int) -> Tuple[Any, List[List[str]]]:
        """Обучает BPE модель"""
        try:
            # Создаем токенизатор
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            # Создаем тренер
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=False,
            )

            # Обучаем на корпусе
            tokenizer.train_from_iterator(self.corpus, trainer=trainer)

            # Токенизируем корпус
            tokenized_texts = []
            for text in self.corpus:
                if text.strip():
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    tokenized_texts.append(tokens)

            return tokenizer, tokenized_texts

        except Exception as e:
            st.error(f"Ошибка обучения BPE: {e}")
            # Возвращаем заглушку
            return None, [[] for _ in self.corpus]

    def train_wordpiece(self, vocab_size: int, min_frequency: int) -> Tuple[Any, List[List[str]]]:
        """Обучает WordPiece модель"""
        try:
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=False,
                continuing_subword_prefix="##"
            )

            tokenizer.train_from_iterator(self.corpus, trainer=trainer)

            tokenized_texts = []
            for text in self.corpus:
                if text.strip():
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    tokenized_texts.append(tokens)

            return tokenizer, tokenized_texts

        except Exception as e:
            st.error(f"Ошибка обучения WordPiece: {e}")
            return None, [[] for _ in self.corpus]

    def train_unigram_sentencepiece(self, vocab_size: int, min_frequency: int) -> Tuple[Any, List[List[str]]]:
        """Обучает Unigram модель используя sentencepiece"""
        try:
            corpus_file = self.prepare_corpus_file()
            
            # Адаптируем размер словаря под размер корпуса
            unique_words = len(set(" ".join(self.corpus).split()))
            actual_vocab_size = min(vocab_size, unique_words * 2, 8000)
            
            if actual_vocab_size < 100:
                st.warning(f"Слишком маленький корпус для SentencePiece. Требуется больше текстов.")
                return None, [[] for _ in self.corpus]

            model_prefix = "unigram_temp_model"

            spm.SentencePieceTrainer.train(
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=actual_vocab_size,
                model_type='unigram',
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece='[PAD]',
                unk_piece='[UNK]',
                split_by_whitespace=True,
                max_sentence_length=10000,
            )

            # Загружаем обученную модель
            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")

            # Токенизируем корпус
            tokenized_texts = []
            for text in self.corpus:
                if text.strip():
                    tokens = sp.encode_as_pieces(text)
                    tokenized_texts.append(tokens)

            # Удаляем временные файлы
            for ext in ['.model', '.vocab']:
                if os.path.exists(f"{model_prefix}{ext}"):
                    os.remove(f"{model_prefix}{ext}")
            if os.path.exists(corpus_file):
                os.remove(corpus_file)

            return sp, tokenized_texts

        except Exception as e:
            st.error(f"Ошибка обучения Unigram SentencePiece: {e}")
            # Удаляем временные файлы в случае ошибки
            for file in ['temp_corpus.txt', 'unigram_temp_model.model', 'unigram_temp_model.vocab']:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
            return None, [[] for _ in self.corpus]

    def train_unigram_huggingface(self, vocab_size: int, min_frequency: int) -> Tuple[Any, List[List[str]]]:
        """Обучает Unigram модель через Hugging Face"""
        try:
            tokenizer = Tokenizer(Unigram())
            tokenizer.pre_tokenizer = Whitespace()

            trainer = UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                unk_token="[UNK]",
                max_piece_length=16,
            )

            tokenizer.train_from_iterator(self.corpus, trainer=trainer)

            tokenized_texts = []
            for text in self.corpus:
                if text.strip():
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    tokenized_texts.append(tokens)

            return tokenizer, tokenized_texts

        except Exception as e:
            st.error(f"Ошибка обучения Unigram HF: {e}")
            return None, [[] for _ in self.corpus]

    def evaluate_model(self, model_name: str, tokenized_texts: List[List[str]],
                      processing_time: float, vocab_size: int) -> Dict:
        """Вычисляет все метрики для модели"""
        # Фильтруем пустые тексты
        valid_original = [text for text in self.corpus if text.strip()]
        valid_tokenized = [tokens for tokens in tokenized_texts if tokens]

        if not valid_tokenized or not valid_original:
            return {
                'model': model_name,
                'vocab_size': vocab_size,
                'actual_vocab_size': 0,
                'fragmentation_rate': 0,
                'compression_ratio': 1,
                'reconstruction_accuracy': 0,
                'processing_time_sec': round(processing_time, 2),
                'avg_token_length': 0,
                'status': 'failed'
            }

        try:
            # Процент фрагментации
            fragmentation = self.calculate_fragmentation(valid_tokenized)

            # Коэффициент сжатия
            compression_ratio = self.calculate_compression_ratio(valid_original, valid_tokenized)

            # Реконструкция текста
            reconstructed_texts = []
            for tokens in valid_tokenized:
                reconstructed = self.reconstruct_text_for_model(tokens, model_name)
                reconstructed_texts.append(reconstructed)

            reconstruction_accuracy = self.calculate_reconstruction_accuracy(valid_original, reconstructed_texts)

            # Статистика по токенам
            all_tokens = [token for tokens in valid_tokenized for token in tokens]
            avg_token_length = np.mean([len(token) for token in all_tokens]) if all_tokens else 0
            actual_vocab_size = len(set(all_tokens))

            return {
                'model': model_name,
                'vocab_size': vocab_size,
                'actual_vocab_size': actual_vocab_size,
                'fragmentation_rate': round(fragmentation, 2),
                'compression_ratio': round(compression_ratio, 3),
                'reconstruction_accuracy': round(reconstruction_accuracy, 2),
                'processing_time_sec': round(processing_time, 2),
                'avg_token_length': round(avg_token_length, 2),
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Ошибка оценки модели {model_name}: {e}")
            return {
                'model': model_name,
                'vocab_size': vocab_size,
                'actual_vocab_size': 0,
                'fragmentation_rate': 0,
                'compression_ratio': 1,
                'reconstruction_accuracy': 0,
                'processing_time_sec': round(processing_time, 2),
                'avg_token_length': 0,
                'status': 'error'
            }

    def run_comparison(self, vocab_sizes: List[int] = None, min_frequency: int = 2, show_debug: bool = False) -> pd.DataFrame:
        """Запускает сравнительный анализ моделей"""
        if not self.corpus:
            st.error("Корпус пуст. Нет данных для анализа.")
            return pd.DataFrame()

        if vocab_sizes is None:
            vocab_sizes = [1000, 2000]

        st.write(f"🔍 **Анализ корпуса:** {len(self.corpus)} текстов, {len(set(' '.join(self.corpus).split()))} уникальных слов")

        models_to_train = [
            ("BPE", self.train_bpe),
            ("WordPiece", self.train_wordpiece), 
            ("Unigram_HF", self.train_unigram_huggingface),
            ("Unigram_SP", self.train_unigram_sentencepiece)
        ]

        progress_bar = st.progress(0)
        total_steps = len(vocab_sizes) * len(models_to_train)
        current_step = 0

        for vocab_size in vocab_sizes:
            st.write(f"### 📊 Размер словаря: {vocab_size}")

            for model_name, train_func in models_to_train:
                # Пропускаем SentencePiece для больших словарей
                if model_name == "Unigram_SP" and vocab_size > 8000:
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    continue

                status_text = st.empty()
                status_text.text(f"Обучается {model_name}...")

                try:
                    start_time = time.time()
                    model, tokens = train_func(vocab_size, min_frequency)
                    processing_time = time.time() - start_time

                    if model is not None and tokens:
                        metrics = self.evaluate_model(model_name, tokens, processing_time, vocab_size)
                        self.results.append(metrics)
                        
                        if show_debug and metrics.get('status') == 'success':
                            self.debug_tokenization(model_name, tokens)
                            
                        status_text.success(f"{model_name} ✓")
                    else:
                        status_text.warning(f"{model_name} не удалось обучить")

                except Exception as e:
                    status_text.error(f"Ошибка {model_name}: {str(e)}")

                current_step += 1
                progress_bar.progress(current_step / total_steps)

        progress_bar.empty()
        
        # Фильтруем успешные результаты
        successful_results = [r for r in self.results if r.get('status') in ['success', None]]
        return pd.DataFrame(successful_results)

def main():
    st.set_page_config(
        page_title="Advanced Text Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Advanced Text Analysis Tool")
    st.markdown("Анализируйте текст с различными методами токенизации и сравнивайте subword модели")
    
    # Initialize processors
    processor = TextProcessor()
    text_cleaner = None
    universal_preprocessor = UniversalPreprocessor()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Parameters")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Sample Dataset", "Custom Text", "Upload File"]
        )
        
        text_data = ""
        corpus = []
        original_text_data = ""  # Сохраняем оригинальный текст
        
        if data_source == "Sample Dataset":
            dataset_choice = st.selectbox(
                "Choose dataset",
                list(SAMPLE_DATASETS.keys()),
                format_func=lambda x: SAMPLE_DATASETS[x]['name']
            )
            if dataset_choice:
                corpus = SAMPLE_DATASETS[dataset_choice]['data']
                text_data = " ".join(corpus)
                original_text_data = text_data
                with st.expander("View Sample Text"):
                    st.text(text_data)
        
        elif data_source == "Custom Text":
            text_data = st.text_area("Enter your text", height=200,
                                   placeholder="Paste your text here...")
            original_text_data = text_data
            if text_data.strip():
                corpus = [text_data]
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file is not None:
                text_data = uploaded_file.getvalue().decode("utf-8")
                original_text_data = text_data
                corpus = [text_data]
                with st.expander("View Uploaded Text"):
                    st.text(text_data[:1000] + "..." if len(text_data) > 1000 else text_data)
        
        # Text preprocessing pipeline
        st.subheader("🔄 Text Preprocessing Pipeline")
        
        # Step 1: Universal Preprocessing
        st.markdown("**1. Universal Preprocessing**")
        enable_universal_preprocessing = st.checkbox("Enable Universal Preprocessing", value=False)
        
        if enable_universal_preprocessing:
            col1, col2 = st.columns(2)
            with col1:
                normalize_punctuation = st.checkbox("Normalize punctuation", value=True)
                normalize_whitespace = st.checkbox("Normalize whitespace", value=True)
                replace_numbers = st.checkbox("Replace numbers", value=True)
                replace_urls = st.checkbox("Replace URLs", value=True)
            with col2:
                replace_emails = st.checkbox("Replace emails", value=True)
                expand_abbreviations = st.checkbox("Expand abbreviations", value=True)
                expand_special_abbr = st.checkbox("Expand special abbreviations", value=True)
                preserve_sentences = st.checkbox("Preserve sentence endings", value=True)
        
        # Step 2: Text Cleaning
        st.markdown("**2. Text Cleaning**")
        enable_cleaning = st.checkbox("Enable Advanced Text Cleaning", value=False)
        
        if enable_cleaning:
            cleaning_language = st.selectbox("Cleaning Language", ["russian", "tatar", "english"])
            clean_html = st.checkbox("Remove HTML tags", value=True)
            remove_special_chars = st.checkbox("Remove special characters", value=True)
            cleaning_lowercase = st.checkbox("Convert to lowercase", value=True)
            cleaning_remove_stopwords = st.checkbox("Remove stopwords", value=True)
        
        # Step 3: Analysis Parameters
        st.subheader("🔍 Analysis Parameters")
        language = st.selectbox("Analysis Language", ["en", "ru", "tt"])
        tokenization = st.selectbox(
            "Tokenization Method",
            ["word", "nltk", "split"]
        )
        normalization = st.selectbox(
            "Normalization",
            ["lowercase", "stemming", "lemmatization", "none"]
        )
        remove_stopwords = st.checkbox("Remove Stop Words (Analysis)")
        
        # Step 4: Advanced Analysis
        st.subheader("🔬 Advanced Analysis")
        enable_subword_analysis = st.checkbox("Enable Subword Model Comparison")
        if enable_subword_analysis:
            st.markdown("**🧠 Subword Models:**")
            col1, col2 = st.columns(2)
            with col1:
                enable_bpe = st.checkbox("BPE", value=True)
                enable_wordpiece = st.checkbox("WordPiece", value=True)
            with col2:
                enable_unigram_hf = st.checkbox("Unigram HF", value=True)
                enable_unigram_sp = st.checkbox("Unigram SP", value=True)
            
            vocab_sizes = st.multiselect(
                "Vocabulary Sizes",
                [500, 1000, 2000, 5000, 10000],
                default=[1000, 2000]
            )
            min_frequency = st.slider("Minimum Frequency", 1, 10, 2)
            show_debug_info = st.checkbox("Show Debug Information", value=False)
        
        analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button and text_data.strip():
        # Apply preprocessing pipeline
        processed_text_data = text_data
        preprocessing_steps = []
        preprocessing_info = ""
        
        # Step 1: Universal Preprocessing
        if enable_universal_preprocessing:
            with st.spinner("Applying universal preprocessing..."):
                universal_preprocessing_config = {
                    'normalize_punctuation': normalize_punctuation,
                    'normalize_whitespace': normalize_whitespace,
                    'replace_numbers': replace_numbers,
                    'replace_urls': replace_urls,
                    'replace_emails': replace_emails,
                    'expand_abbreviations': expand_abbreviations,
                    'expand_special_abbreviations': expand_special_abbr,
                    'preserve_sentence_endings': preserve_sentences
                }
                
                processed_text_data = universal_preprocessor.preprocess_text(
                    processed_text_data, **universal_preprocessing_config
                )
                preprocessing_steps.append("Universal Preprocessing")
        
        # Step 2: Text Cleaning
        if enable_cleaning:
            with st.spinner("Cleaning text..."):
                text_cleaner = TextCleaner(
                    lowercase=cleaning_lowercase,
                    remove_stopwords=cleaning_remove_stopwords,
                    language=cleaning_language
                )
                
                processed_text_data = text_cleaner.clean_text(
                    processed_text_data,
                    clean_html=clean_html,
                    remove_special_chars=remove_special_chars,
                    normalize_whitespace=True
                )
                preprocessing_steps.append("Text Cleaning")
        
        # Calculate preprocessing statistics
        original_length = len(text_data.split())
        processed_length = len(processed_text_data.split())
        removed_percentage = ((original_length - processed_length) / original_length * 100) if original_length > 0 else 0
        
        preprocessing_info = f"""
        **Preprocessing Pipeline Results:**
        - **Steps applied:** {', '.join(preprocessing_steps) if preprocessing_steps else 'None'}
        - **Original words:** {original_length}
        - **After preprocessing:** {processed_length}
        - **Removed:** {original_length - processed_length} words ({removed_percentage:.1f}%)
        """
        
        # Basic text analysis
        with st.spinner("Performing basic text analysis..."):
            tokens = processor.tokenize(processed_text_data, tokenization, language)
            tokens = processor.normalize(tokens, normalization)
            if remove_stopwords:
                tokens = processor.remove_stopwords(tokens, language)
            
            analysis = generate_analysis(tokens, processed_text_data, language)
            display_basic_results(analysis, original_text_data, processed_text_data, tokens, language, 
                               preprocessing_steps, preprocessing_info)
        
        # Subword model comparison
        if enable_subword_analysis and corpus:
            st.header("🔬 Subword Model Comparison")
            
            # Use processed corpus as list of texts, not single string
            if isinstance(processed_text_data, str):
                analysis_corpus = [processed_text_data]
            else:
                analysis_corpus = processed_text_data
            
            # Добавьте проверку размера корпуса
            if len(analysis_corpus) < 2:
                st.warning("""
                **Рекомендация:** Для лучшего сравнения subword моделей добавьте больше текстов.
                Subword модели требуют достаточного количества данных для эффективного обучения.
                """)
            
            with st.expander("ℹ️ Информация о моделях"):
                st.markdown("""
                **Доступные модели:**
                - **🧩 BPE (Byte Pair Encoding)** - популярный алгоритм для субсловной токенизации
                - **🔤 WordPiece** - используется в BERT, похож на BPE но с другим критерием выбора  
                - **📊 Unigram HF** - вероятностная модель через Hugging Face
                - **🎯 Unigram SP** - реализация Unigram с поддержкой Unicode через SentencePiece
                
                **Рекомендации по использованию:**
                - Для маленьких корпусов (< 1000 слов) используйте размер словаря 500-1000
                - Для средних корпусов (1000-5000 слов) используйте 1000-2000
                - Для больших корпусов (> 5000 слов) можно использовать 3000-5000
                """)
            
            comparator = SubwordModelComparator(analysis_corpus)
            
            with st.spinner("Training and comparing subword models..."):
                # Фильтруем выбранные модели
                selected_models = []
                if enable_bpe:
                    selected_models.append(("BPE", comparator.train_bpe))
                if enable_wordpiece:
                    selected_models.append(("WordPiece", comparator.train_wordpiece))
                if enable_unigram_hf:
                    selected_models.append(("Unigram_HF", comparator.train_unigram_huggingface))
                if enable_unigram_sp:
                    selected_models.append(("Unigram_SP", comparator.train_unigram_sentencepiece))
                
                if not selected_models:
                    st.warning("Пожалуйста, выберите хотя бы одну модель для сравнения.")
                else:
                    # Переопределяем метод run_comparison для выбранных моделей
                    results_df = run_custom_comparison(
                        comparator, selected_models, vocab_sizes, min_frequency, show_debug_info
                    )
                    
                    if not results_df.empty:
                        display_subword_results(results_df)
                    else:
                        st.error("Не удалось обучить ни одну модель. Попробуйте увеличить объем текста или изменить параметры.")
    
    elif analyze_button and not text_data.strip():
        st.warning("Please provide some text to analyze!")
    
    else:
        show_welcome_message()

def run_custom_comparison(comparator, selected_models, vocab_sizes, min_frequency, show_debug):
    """Запускает сравнение только для выбранных моделей"""
    if not vocab_sizes:
        vocab_sizes = [1000, 2000]

    progress_bar = st.progress(0)
    total_steps = len(vocab_sizes) * len(selected_models)
    current_step = 0

    for vocab_size in vocab_sizes:
        st.write(f"### 📊 Размер словаря: {vocab_size}")

        for model_name, train_func in selected_models:
            # Пропускаем SentencePiece для больших словарей
            if model_name == "Unigram_SP" and vocab_size > 8000:
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                continue

            status_text = st.empty()
            status_text.text(f"Обучается {model_name}...")

            try:
                start_time = time.time()
                model, tokens = train_func(vocab_size, min_frequency)
                processing_time = time.time() - start_time

                if model is not None and tokens:
                    metrics = comparator.evaluate_model(model_name, tokens, processing_time, vocab_size)
                    comparator.results.append(metrics)
                    
                    if show_debug and metrics.get('status') == 'success':
                        comparator.debug_tokenization(model_name, tokens)
                        
                    status_text.success(f"{model_name} ✓")
                else:
                    status_text.warning(f"{model_name} не удалось обучить")

            except Exception as e:
                status_text.error(f"Ошибка {model_name}: {str(e)}")

            current_step += 1
            progress_bar.progress(current_step / total_steps)

    progress_bar.empty()
    
    # Фильтруем успешные результаты
    successful_results = [r for r in comparator.results if r.get('status') in ['success', None]]
    return pd.DataFrame(successful_results)

def generate_analysis(tokens, original_text, language):
    """Generate comprehensive text analysis"""
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    avg_token_length = sum(len(token) for token in tokens) / total_tokens if total_tokens > 0 else 0
    
    token_freq = Counter(tokens)
    
    # OOV analysis
    if language == 'en':
        try:
            common_vocab = set(nltk.corpus.words.words()[:5000])
        except:
            common_vocab = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with'])
    elif language == 'ru':
        common_vocab = set(['в', 'на', 'и', 'с', 'по', 'к', 'у', 'о', 'не', 'что'])
    elif language == 'tt':
        # Базовый словарь татарских слов
        common_vocab = set(['һәм', 'вә', 'белән', 'өчен', 'әле', 'инде', 'бик', 'үк', 'күп', 'аз',
                           'бар', 'юк', 'тел', 'халык', 'мәдәният', 'тарих', 'шәһәр', 'Казан'])
    else:
        common_vocab = set()
    
    oov_tokens = [token for token in tokens if token not in common_vocab]
    oov_ratio = len(oov_tokens) / total_tokens if total_tokens > 0 else 0
    
    sentences = nltk.sent_tokenize(original_text)
    avg_sentence_length = sum(len(nltk.word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    
    return {
        'statistics': {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'avg_token_length': round(avg_token_length, 2),
            'oov_ratio': round(oov_ratio, 4),
            'vocabulary_richness': round(unique_tokens / total_tokens, 4) if total_tokens > 0 else 0,
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2)
        },
        'token_freq': token_freq,
        'tokens': tokens,
        'sentences': sentences
    }

def display_basic_results(analysis, original_text, processed_text, tokens, language, 
                         preprocessing_steps, preprocessing_info):
    """Display basic text analysis results"""
    st.header("📈 Text Analysis Results")
    
    # Show preprocessing results
    if preprocessing_steps:
        with st.expander("🔄 Preprocessing Pipeline Summary", expanded=True):
            st.markdown(preprocessing_info)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text Sample")
                st.text_area("", original_text[:500] + "..." if len(original_text) > 500 else original_text, 
                           height=150, key="original_preview")
            with col2:
                st.subheader("Processed Text Sample")
                st.text_area("", processed_text[:500] + "..." if len(processed_text) > 500 else processed_text, 
                           height=150, key="processed_preview")
    
    # Language info
    language_names = {'en': 'English', 'ru': 'Russian', 'tt': 'Tatar'}
    st.info(f"**Analyzed language:** {language_names.get(language, language)}")
    
    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = analysis['statistics']
    
    with col1:
        st.metric("Total Tokens", stats['total_tokens'])
    with col2:
        st.metric("Unique Tokens", stats['unique_tokens'])
    with col3:
        st.metric("Avg Token Length", stats['avg_token_length'])
    with col4:
        st.metric("OOV Ratio", f"{stats['oov_ratio'] * 100:.2f}%")
    with col5:
        st.metric("Vocabulary Richness", f"{stats['vocabulary_richness'] * 100:.2f}%")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📏 Token Length", "📈 Frequency", "📋 Data"])
    
    with tab1:
        display_length_analysis(tokens)
    
    with tab2:
        display_frequency_analysis(analysis['token_freq'], language)
    
    with tab3:
        display_sample_data(original_text, processed_text, tokens, analysis['sentences'], bool(preprocessing_steps))

def display_subword_results(results_df):
    """Display subword model comparison results"""
    st.header("🎯 Subword Model Comparison Results")
    
    # Results table
    st.subheader("Detailed Results")
    
    # Добавляем цветовое кодирование для лучшей визуализации
    styled_df = results_df.style.format({
        'fragmentation_rate': '{:.2f}%',
        'compression_ratio': '{:.3f}',
        'reconstruction_accuracy': '{:.2f}%',
        'processing_time_sec': '{:.2f}s',
        'avg_token_length': '{:.2f}'
    }).background_gradient(subset=['fragmentation_rate'], cmap='Reds_r')\
      .background_gradient(subset=['reconstruction_accuracy'], cmap='Greens')\
      .background_gradient(subset=['compression_ratio'], cmap='Blues')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Визуализации
    if len(results_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fragmentation rate comparison
            fig_frag = px.bar(results_df, x='model', y='fragmentation_rate', 
                             color='vocab_size', barmode='group',
                             title='📊 Fragmentation Rate by Model',
                             labels={'fragmentation_rate': 'Fragmentation Rate (%)', 'model': 'Model'})
            fig_frag.update_layout(template="plotly_white")
            st.plotly_chart(fig_frag, use_container_width=True)
        
        with col2:
            # Compression ratio comparison
            fig_comp = px.bar(results_df, x='model', y='compression_ratio',
                             color='vocab_size', barmode='group',
                             title='📈 Compression Ratio by Model',
                             labels={'compression_ratio': 'Compression Ratio', 'model': 'Model'})
            fig_comp.update_layout(template="plotly_white")
            st.plotly_chart(fig_comp, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Reconstruction accuracy
            fig_acc = px.bar(results_df, x='model', y='reconstruction_accuracy',
                            color='vocab_size', barmode='group',
                            title='🎯 Reconstruction Accuracy by Model',
                            labels={'reconstruction_accuracy': 'Accuracy (%)', 'model': 'Model'})
            fig_acc.update_layout(template="plotly_white")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col4:
            # Processing time
            fig_time = px.bar(results_df, x='model', y='processing_time_sec',
                             color='vocab_size', barmode='group',
                             title='⏱️ Processing Time by Model',
                             labels={'processing_time_sec': 'Time (seconds)', 'model': 'Model'})
            fig_time.update_layout(template="plotly_white")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Best models analysis
        st.subheader("🏆 Best Performing Models")
        
        successful_models = results_df[results_df['actual_vocab_size'] > 50]
        
        if not successful_models.empty:
            best_fragmentation = successful_models.loc[successful_models['fragmentation_rate'].idxmin()]
            best_compression = successful_models.loc[successful_models['compression_ratio'].idxmin()]
            best_reconstruction = successful_models.loc[successful_models['reconstruction_accuracy'].idxmax()]
            best_speed = successful_models.loc[successful_models['processing_time_sec'].idxmin()]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Fragmentation", 
                         f"{best_fragmentation['model']}", 
                         f"{best_fragmentation['fragmentation_rate']}%")
            with col2:
                st.metric("Best Compression", 
                         f"{best_compression['model']}",
                         f"{best_compression['compression_ratio']}")
            with col3:
                st.metric("Best Reconstruction", 
                         f"{best_reconstruction['model']}",
                         f"{best_reconstruction['reconstruction_accuracy']}%")
            with col4:
                st.metric("Fastest", 
                         f"{best_speed['model']}",
                         f"{best_speed['processing_time_sec']}s")

def display_length_analysis(tokens):
    """Display token length analysis"""
    token_lengths = [len(token) for token in tokens]
    if token_lengths:
        fig_length = px.histogram(
            x=token_lengths,
            title="Distribution of Token Lengths",
            labels={'x': 'Token Length', 'y': 'Frequency'},
            nbins=min(20, len(set(token_lengths)))
        )
        fig_length.update_layout(template="plotly_white")
        st.plotly_chart(fig_length, use_container_width=True)

def display_frequency_analysis(token_freq, language):
    """Display token frequency analysis"""
    top_tokens = token_freq.most_common(20)
    if top_tokens:
        tokens_list, counts_list = zip(*top_tokens)
        df_freq = pd.DataFrame({
            'Token': tokens_list,
            'Frequency': counts_list
        })
        
        fig_freq = px.bar(
            df_freq,
            x='Token',
            y='Frequency',
            title=f"Top 20 Most Frequent Tokens ({language})"
        )
        fig_freq.update_layout(xaxis_tickangle=-45, template="plotly_white")
        st.plotly_chart(fig_freq, use_container_width=True)

def display_sample_data(original_text, processed_text, tokens, sentences, preprocessing_enabled=False):
    """Display sample data"""
    col1, col2 = st.columns(2)
    
    with col1:
        if preprocessing_enabled:
            st.subheader("Original Text Sample")
            st.text_area("", original_text[:800] + "..." if len(original_text) > 800 else original_text, 
                       height=200, key="original_text")
        else:
            st.subheader("Text Sample")
            st.text_area("", original_text[:800] + "..." if len(original_text) > 800 else original_text, 
                       height=200, key="original_text")
    
    with col2:
        st.subheader("Processed Tokens")
        st.text_area("", ", ".join(tokens[:100]), height=200, key="tokens_preview")
        
        st.subheader("Token Statistics")
        token_lengths = [len(token) for token in tokens]
        if token_lengths:
            stats_df = pd.DataFrame({
                'Statistic': ['Min Length', 'Max Length', 'Median Length', 'Std Dev'],
                'Value': [
                    min(token_lengths),
                    max(token_lengths),
                    np.median(token_lengths),
                    np.std(token_lengths)
                ]
            })
            st.dataframe(stats_df, use_container_width=True)

def show_welcome_message():
    """Display welcome message"""
    st.markdown("""
    ## 🚀 Welcome to Advanced Text Analysis Tool!
    
    This tool combines traditional NLP analysis with advanced preprocessing and subword model comparison.
    
    ### 📝 To get started:
    1. **Select your data source** in the sidebar
    2. **Configure preprocessing pipeline**:
       - **Universal Preprocessing**: Normalize punctuation, replace entities, expand abbreviations
       - **Text Cleaning**: Remove HTML, special characters, stopwords
    3. **Configure analysis parameters**:
       - Language (English, Russian, or Tatar)
       - Tokenization method and normalization
    4. **Enable advanced analysis** for subword model comparison
    5. **Click 'Analyze Text'** to see comprehensive results!
    
    ### 🛠️ Preprocessing Features:
    - **Universal Preprocessing**: Standardize text format, replace numbers/URLs/emails with tokens
    - **Text Cleaning**: Remove HTML tags, special characters, and stopwords
    - **Multi-language Support**: English, Russian, and Tatar text processing
    - **Flexible Pipeline**: Configure each preprocessing step independently
    
    ### 🔬 Advanced Features:
    - **Subword Model Training**: Train BPE, WordPiece, Unigram models
    - **Comparative Analysis**: Compare performance across different vocabulary sizes
    - **Quality Metrics**: Fragmentation rate, compression ratio, reconstruction accuracy
    
    ### 🎯 Use Cases:
    - Preprocessing text for machine learning pipelines
    - Choosing optimal tokenization strategies
    - Understanding subword model trade-offs
    - Educational purposes in computational linguistics
    - Research and development of NLP systems
    """)

if __name__ == "__main__":
    main()