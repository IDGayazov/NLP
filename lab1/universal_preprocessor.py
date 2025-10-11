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
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

# Subword модели
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm

class UniversalPreprocessor:
    """
    Универсальный препроцессор для приведения текста к единому стандарту
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация препроцессора

        Args:
            config_path: путь к JSON файлу с конфигурацией
        """
        self.default_config = {
            # Стандартизация пунктуации
            'normalize_punctuation': True,
            'normalize_whitespace': True,

            # Замена на токены
            'replace_numbers': True,
            'replace_urls': True,
            'replace_emails': True,
            'replace_currencies': True,
            'replace_phone_numbers': True,

            # Обработка сокращений
            'expand_abbreviations': True,
            'expand_special_abbreviations': True,

            # Дополнительные настройки
            'preserve_sentence_endings': True,
            'remove_extra_spaces': True
        }

        # Загружаем конфигурацию
        self.config = self.default_config.copy()
        if config_path:
            self.load_config(config_path)

        # Инициализируем правила
        self._init_patterns()
        self._init_abbreviations()

    def load_config(self, config_path: str) -> None:
        """Загрузка конфигурации из JSON файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            self.config.update(user_config)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}. Используются настройки по умолчанию.")

    def save_config(self, config_path: str) -> None:
        """Сохранение текущей конфигурации в файл"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def _init_patterns(self) -> None:
        """Инициализация регулярных выражений"""

        # Числительные
        self.number_patterns = [
            # Целые числа с разделителями
            (r'\b\d{1,3}(?:[ ,]\d{3})+\b', '<NUM>'),  # 1,000, 10 000
            # Десятичные дроби
            (r'\b\d+[.,]\d+\b', '<NUM>'),  # 3.14, 2,5
            # Простые числа
            (r'\b\d+\b', '<NUM>'),  # 123, 45
        ]

        # URL и email
        self.url_pattern = (r'https?://[^\s]+|www\.[^\s]+', '<URL>')
        self.email_pattern = (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>')

        # Номера телефонов
        self.phone_patterns = [
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE>'),  # 123-456-7890
            (r'\b\d{1,2}[-.]?\d{3}[-.]?\d{2}[-.]?\d{2}\b', '<PHONE>'),  # 8-912-34-56
            (r'\b\+?[\d\s\-\(\)]{7,}\b', '<PHONE>'),  # Международные форматы
        ]

        # Валюты
        self.currency_pattern = (r'\b\d+[.,]?\d*\s*[₽$€£¥]\b|\b[₽$€£¥]\s*\d+[.,]?\d*\b', '<CURRENCY>')

        # Пунктуация
        self.punctuation_replacements = {
            '…': '...',
            '«': '"',
            '»': '"',
            '„': '"',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '–': '-',
            '—': '-',
        }

    def _init_abbreviations(self) -> None:
        """Инициализация словаря сокращений"""

        # Общеязыковые сокращения (русский)
        self.common_abbreviations = {
            'т.е.': 'то есть',
            'т.д.': 'так далее',
            'т.п.': 'тому подобное',
            'и т.д.': 'и так далее',
            'и т.п.': 'и тому подобное',
            'и др.': 'и другие',
            'и пр.': 'и прочие',
            'т.к.': 'так как',
            'т.н.': 'так называемый',
            'т.о.': 'таким образом',
            'с.г.': 'сего года',
            'н.э.': 'нашей эры',
            'до н.э.': 'до нашей эры',
            'г.': 'год',
            'гг.': 'годы',
            'вв.': 'века',
            'см.': 'смотри',
            'стр.': 'страница',
            'рис.': 'рисунок',
            'напр.': 'например',
            'мин.': 'минут',
            'сек.': 'секунд',
            'ч.': 'час',
            'кг.': 'килограмм',
            'см.': 'сантиметр',
            'м.': 'метр',
            'км.': 'километр',
            'руб.': 'рубль',
            'долл.': 'доллар',
            'евро.': 'евро',
        }

        # Специальные сокращения (можно расширить)
        self.special_abbreviations = {
            'США': 'Соединенные Штаты Америки',
            'РФ': 'Российская Федерация',
            'СССР': 'Союз Советских Социалистических Республик',
            'ООН': 'Организация Объединенных Наций',
            'НАТО': 'Организация Североатлантического договора',
        }

    def normalize_punctuation(self, text: str) -> str:
        """Стандартизация пунктуации"""
        if not self.config['normalize_punctuation']:
            return text

        for old, new in self.punctuation_replacements.items():
            text = text.replace(old, new)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Стандартизация пробельных символов"""
        if not self.config['normalize_whitespace']:
            return text

        # Заменяем все пробельные символы на обычные пробелы
        text = re.sub(r'\s+', ' ', text)

        # Убираем пробелы вокруг пунктуации
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        text = re.sub(r'\s+([)])', r'\1', text)

        # Добавляем пробелы после пунктуации, если нужно
        if self.config['preserve_sentence_endings']:
            text = re.sub(r'([.!?])([А-ЯA-Z])', r'\1 \2', text)

        return text.strip()

    def replace_with_tokens(self, text: str) -> str:
        """Замена числительных, URL, email на токены"""

        # URL
        if self.config['replace_urls']:
            pattern, replacement = self.url_pattern
            text = re.sub(pattern, replacement, text)

        # Email
        if self.config['replace_emails']:
            pattern, replacement = self.email_pattern
            text = re.sub(pattern, replacement, text)

        # Числительные
        if self.config['replace_numbers']:
            for pattern, replacement in self.number_patterns:
                text = re.sub(pattern, replacement, text)

        # Номера телефонов
        if self.config['replace_phone_numbers']:
            for pattern, replacement in self.phone_patterns:
                text = re.sub(pattern, replacement, text)

        # Валюты
        if self.config['replace_currencies']:
            pattern, replacement = self.currency_pattern
            text = re.sub(pattern, replacement, text)

        return text

    def expand_abbreviations(self, text: str) -> str:
        """Раскрытие сокращений"""
        if not self.config['expand_abbreviations']:
            return text

        # Общеязыковые сокращения
        for abbrev, expansion in self.common_abbreviations.items():
            # Используем границы слова для точного совпадения
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

        # Специальные сокращения
        if self.config['expand_special_abbreviations']:
            for abbrev, expansion in self.special_abbreviations.items():
                pattern = r'\b' + re.escape(abbrev) + r'\b'
                text = re.sub(pattern, expansion, text)

        return text

    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        Основная функция предобработки текста

        Args:
            text: исходный текст
            **kwargs: временные настройки (переопределяют конфигурацию)

        Returns:
            обработанный текст
        """
        if not text:
            return ""

        # Сохраняем оригинальную конфигурацию
        original_config = self.config.copy()

        try:
            # Временно применяем переданные настройки
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value

            # Применяем все этапы обработки
            processed_text = text

            # 1. Стандартизация пунктуации
            processed_text = self.normalize_punctuation(processed_text)

            # 2. Замена на токены
            processed_text = self.replace_with_tokens(processed_text)

            # 3. Раскрытие сокращений
            processed_text = self.expand_abbreviations(processed_text)

            # 4. Стандартизация пробелов
            processed_text = self.normalize_whitespace(processed_text)

            return processed_text

        finally:
            # Восстанавливаем оригинальную конфигурацию
            self.config = original_config

    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Пакетная обработка списка текстов"""
        return [self.preprocess_text(text, **kwargs) for text in texts]