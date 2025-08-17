#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konfiguracja dla modelowania tematycznego
"""

import os
from pathlib import Path

# Ścieżki
DATABASE_PATH = os.getenv("TOPIC_DATABASE_PATH", "../../data/databases/analysis_forums.db")
OUTPUT_DIR = Path(os.getenv("TOPIC_OUTPUT_DIR", "../../data/topics"))

# Parametry analizy
ANALYSIS_PARAMS = {
    'min_tokens': 25,           # Minimalna liczba tokenów w poście
    'max_tokens': 1000,         # Maksymalna liczba tokenów w poście
    'max_posts_per_user': 100,  # Maksymalna liczba postów na użytkownika
    'min_word_count': 5,        # Minimalna liczba wystąpień słowa w Top2Vec
    'topic_merge_delta': 0.1,   # Parametr łączenia tematów
    'topic_merge_threshold': 0.1,  # Próg łączenia tematów
    'max_words_per_topic': 20,  # Maksymalna liczba słów kluczowych na temat
    'random_seed': 42           # Ziarno losowości dla powtarzalności wyników
}

# Fora do analizy
_forums_env = os.getenv("TOPIC_FORUMS")
if _forums_env:
    FORUMS = [f.strip() for f in _forums_env.split(',') if f.strip()]
else:
    FORUMS = ['wiara']  # Domyślnie forum 'wiara'

_genders_env = os.getenv("TOPIC_GENDERS")
if _genders_env:
    GENDERS = [g.strip() for g in _genders_env.split(',') if g.strip()]
else:
    GENDERS = ['M', 'K']  # M (male), K (female)

# Parametry wizualizacji
VISUALIZATION_PARAMS = {
    'wordcloud': {
        'width': 800,
        'height': 600,
        'background_color': 'white',
        'max_words': 50,
        'colormap': 'viridis'
    },
    'charts': {
        'figure_size': (12, 8),
        'dpi': 300,
        'bar_colors': {
            'distribution': 'skyblue',
            'scores': 'lightcoral'
        }
    }
}

# Parametry logowania
LOGGING_PARAMS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'encoding': 'utf-8'
}

# Parametry bazy danych
DATABASE_PARAMS = {
    'timeout': 30,
    'check_same_thread': False
}
