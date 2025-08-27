#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do modelowania tematycznego z biblioteką Top2Vec
Analizuje dane z forów radio_katolik i dolina_modlitwy
Osobno dla mężczyzn i kobiet
"""

import sqlite3
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Top2Vec i biblioteki do analizy
from top2vec import Top2Vec
from config import FORUM_CODES
from config import EXCLUDED_SECTIONS
from config import TOPICS_PARAMS
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Konfiguracja logowania
def setup_logging(log_dir: Path, run_slug: str) -> logging.Logger:
    """Konfiguruje logowanie dla konkretnego uruchomienia (slug)."""
    # Utwórz katalog na logi dopiero przy starcie zapisu
    log_dir.mkdir(parents=True, exist_ok=True)
    # Użyj tylko czasu, data jest częścią slug-a
    log_file = log_dir / f"{run_slug}_{datetime.now().strftime('%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(run_slug)

class TopicModelingAnalyzer:
    """Klasa do analizy modelowania tematycznego"""
    
    def __init__(self, db_path: str, output_dir: Path):
        self.db_path = db_path
        self.output_dir = output_dir
        self.logger = None
        
        # Parametry analizy
        self.min_tokens = 25
        self.max_tokens = 1000
        self.max_posts_per_user = 100
        self.random_seed = 42

        # Parametry Top2Vec
        self.embedding_model = TOPICS_PARAMS.get('embedding_model', "doc2vec")
        self.keep_documents_in_model = bool(TOPICS_PARAMS.get('keep_documents_in_model', True))
        self.target_num_topics = TOPICS_PARAMS.get('target_num_topics')
        # Bieżący identyfikator uruchomienia (kod_forum[_kod_forum2]_GENDER_YYYYMMDD)
        self.run_slug: Optional[str] = None
        self.current_run_time: Optional[datetime] = None

    def _get_run_components(self, forums: List[str] | str, gender: str) -> Dict[str, Path | str]:
        when = self.current_run_time or datetime.now()
        date_str = when.strftime('%Y%m%d')
        time_str = when.strftime('%H%M%S')
        gender_dir = 'KM' if str(gender).upper() == 'KM' else str(gender)
        if isinstance(forums, str):
            forum_names = [forums]
        else:
            forum_names = list(forums)
        if set(forum_names) == set(FORUM_CODES.keys()):
            forums_code = 'ALL'
        else:
            forums_code = "_".join(self._forum_to_code(f) for f in forum_names)
        base = self.output_dir
        results_dir = base / "results" / date_str / gender_dir / forums_code / time_str
        models_dir = base / "models" / date_str / gender_dir / forums_code / time_str
        logs_dir = base / "logs" / date_str / gender_dir / forums_code / time_str
        return {
            'date_str': date_str,
            'time_str': time_str,
            'gender_dir': gender_dir,
            'forums_code': forums_code,
            'results_dir': results_dir,
            'models_dir': models_dir,
            'logs_dir': logs_dir,
        }

    def _reduce_to_target_topics(self, model: Top2Vec) -> Top2Vec:
        """Próbuje zredukować liczbę tematów różnymi wariantami API Top2Vec.
        Zwraca model (być może nowy) po redukcji lub model oryginalny jeśli się nie uda.
        """
        try:
            target = self.target_num_topics
            if not (isinstance(target, int) and target >= 2):
                return model
            current = getattr(model, 'get_num_topics', lambda: None)()
            if current is None or target >= current:
                return model

            self.logger.info(f"Redukuję liczbę tematów: {current} -> {target}")

            def _maybe_take_model(result):
                # Jeżeli redukcja zwraca nowy model zamiast modyfikować in-place
                if result is None:
                    return None
                if hasattr(result, 'get_num_topics'):
                    return result
                if isinstance(result, (list, tuple)):
                    for elem in result:
                        if hasattr(elem, 'get_num_topics'):
                            return elem
                if isinstance(result, dict):
                    for _k, _v in result.items():
                        if hasattr(_v, 'get_num_topics'):
                            return _v
                return None

            # 1) klasyczna metoda
            if hasattr(model, 'hierarchical_topic_reduction'):
                reduced_result = None
                try:
                    reduced_result = model.hierarchical_topic_reduction(target)
                except TypeError:
                    reduced_result = model.hierarchical_topic_reduction(num_topics=target)
                # Jeśli nie zadziałało in-place, spróbuj przejąć model ze zwrotki
                new_model = _maybe_take_model(reduced_result)
                if new_model is not None:
                    model = new_model

            # 2) alternatywne nazwy/metody (na wypadek forka)
            if getattr(model, 'get_num_topics', lambda: current)() >= current:
                for alt_name in ['reduce_topics', 'topic_reduction', 'reduce_num_topics']:
                    if hasattr(model, alt_name):
                        try:
                            reduced_result = getattr(model, alt_name)(target)
                        except TypeError:
                            reduced_result = getattr(model, alt_name)(num_topics=target)
                        new_model = _maybe_take_model(reduced_result)
                        if new_model is not None:
                            model = new_model
                        # przerwij pętlę jeśli faktycznie spadła liczba tematów
                        if getattr(model, 'get_num_topics', lambda: current)() < current:
                            break

            self.logger.info(f"Po redukcji liczba tematów: {model.get_num_topics()}")
            if model.get_num_topics() >= current:
                self.logger.warning("Wygląda na to, że redukcja nie zmniejszyła liczby tematów – sprawdź wersję Top2Vec/forka.")
            return model
        except Exception as e:
            self.logger.warning(f"Błąd próby redukcji liczby tematów do {self.target_num_topics}: {e}")
            return model

    def _call_with_reduced(self, func, *args, prefer_reduced: bool = False, **kwargs):
        """Próbuje wywołać metodę z parametrem reduced=True, jeśli jest dostępny i preferowany.
        W przeciwnym wypadku wywołuje bez tego parametru.
        """
        if not callable(func):
            raise TypeError("func must be callable")
        if prefer_reduced:
            try:
                return func(*args, reduced=True, **kwargs)
            except TypeError:
                return func(*args, **kwargs)
        return func(*args, **kwargs)


    def _collect_run_params(self, forums: List[str], gender: str, is_combined: bool) -> Dict:
        """Zbiera parametry wejściowe uruchomienia do zapisania w wynikach."""
        return {
            'forums': list(forums),
            'gender': gender,
            'combined': bool(is_combined),
            'random_seed': int(self.random_seed),
            'min_tokens': int(self.min_tokens),
            'max_tokens': int(self.max_tokens),
            'max_posts_per_user': int(self.max_posts_per_user),
            'excluded_sections': list(EXCLUDED_SECTIONS or []),
            'embedding_model': str(self.embedding_model),
            'keep_documents_in_model': bool(self.keep_documents_in_model),
            'database_path': str(self.db_path),
            'target_num_topics': int(self.target_num_topics) if isinstance(self.target_num_topics, int) else None,
        }

    # Pomocnicze: mapowanie nazwy forum do kodu
    def _forum_to_code(self, forum_name: str) -> str:
        code = FORUM_CODES.get(forum_name)
        if code:
            return code
        # Fallback: znormalizuj nazwę do A-Z0-9 i podkreśleń
        sanitized = re.sub(r'[^A-Za-z0-9]+', '_', forum_name).strip('_')
        return sanitized.upper()[:8] if sanitized else 'FORUM'

    # Pomocnicze: budowanie slug-a dla uruchomienia
    def _build_run_slug(self, forums: List[str] | str, gender: str, when: Optional[datetime] = None) -> str:
        if when is None:
            when = datetime.now()
        if isinstance(forums, str):
            codes_part = self._forum_to_code(forums)
        else:
            codes_part = "_".join(self._forum_to_code(f) for f in forums)
        date_part = when.strftime('%Y%m%d_%H%M%S')
        return f"{codes_part}_{gender}_{date_part}"
        
    def connect_to_db(self) -> sqlite3.Connection:
        """Nawiązuje połączenie z bazą danych"""
        try:
            conn = sqlite3.connect(self.db_path)
            self.logger.info(f"Połączono z bazą danych: {self.db_path}")
            return conn
        except Exception as e:
            self.logger.error(f"Błąd połączenia z bazą danych: {e}")
            raise

    def _gender_predictions_available(self, conn: sqlite3.Connection) -> bool:
        """Sprawdza czy tabela gender_predictions istnieje."""
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gender_predictions'")
            return cur.fetchone() is not None
        except Exception:
            return False
    
    def get_forum_data(self, conn: sqlite3.Connection, forum_name: str, gender: str) -> pd.DataFrame:
        """Pobiera dane z konkretnego forum i płci"""
        self.logger.info(f"Pobieranie danych dla forum: {forum_name}, płeć: {gender}")
        
        # Przygotuj filtr sekcji (case-insensitive)
        excluded_titles = [t.lower() for t in (EXCLUDED_SECTIONS or [])]
        excluded_clause = ""
        params_tail = [self.min_tokens, self.max_tokens]
        if excluded_titles:
            placeholders = ",".join(["?"] * len(excluded_titles))
            excluded_clause = f" AND LOWER(fs.title) NOT IN ({placeholders})"
            # dodamy te parametry na końcu listy

        gender_all = str(gender).upper() == 'KM'
        if self._gender_predictions_available(conn):
            query = """
            SELECT 
                fp.id,
                fp.user_id AS user_id,
                fp.content,
                fp.post_date,
                fp.username,
                fu.gender AS db_gender,
                COALESCE(fu.gender, gp.predicted_gender) AS eff_gender,
                fu.religion,
                fu.localization,
                ft.title as thread_title,
                f.spider_name,
                ta.token_count AS token_count
            FROM forum_posts fp
            JOIN forum_threads ft ON fp.thread_id = ft.id
            JOIN forum_sections fs ON ft.section_id = fs.id
            JOIN forums f ON fs.forum_id = f.id
            LEFT JOIN forum_users fu ON fp.user_id = fu.id
            LEFT JOIN (
                SELECT gp1.user_id, gp1.predicted_gender
                FROM gender_predictions gp1
                JOIN (
                    SELECT user_id, MAX(COALESCE(updated_at, created_at)) AS ts
                    FROM gender_predictions
                    GROUP BY user_id
                ) last ON last.user_id = gp1.user_id AND COALESCE(gp1.updated_at, gp1.created_at) = last.ts
            ) gp ON gp.user_id = fu.id
            JOIN token_analysis ta ON ta.post_id = fp.id
            WHERE f.spider_name = ?
              {gender_filter}
              AND ta.token_count BETWEEN ? AND ?
              {excluded_clause}
              AND fp.content IS NOT NULL
              AND LENGTH(fp.content) > 0
            """
        else:
            query = """
            SELECT 
                fp.id,
                fp.user_id AS user_id,
                fp.content,
                fp.post_date,
                fp.username,
                fu.gender AS db_gender,
                fu.gender AS eff_gender,
                fu.religion,
                fu.localization,
                ft.title as thread_title,
                f.spider_name,
                ta.token_count AS token_count
            FROM forum_posts fp
            JOIN forum_threads ft ON fp.thread_id = ft.id
            JOIN forum_sections fs ON ft.section_id = fs.id
            JOIN forums f ON fs.forum_id = f.id
            LEFT JOIN forum_users fu ON fp.user_id = fu.id
            JOIN token_analysis ta ON ta.post_id = fp.id
            WHERE f.spider_name = ?
              {gender_filter}
              AND ta.token_count BETWEEN ? AND ?
              {excluded_clause}
              AND fp.content IS NOT NULL
              AND LENGTH(fp.content) > 0
            """
        
        try:
            if gender_all:
                gf = ""
                params = [forum_name, self.min_tokens, self.max_tokens]
            else:
                gf = "AND COALESCE(fu.gender, gp.predicted_gender) = ?" if self._gender_predictions_available(conn) else "AND fu.gender = ?"
                params = [forum_name, gender, self.min_tokens, self.max_tokens]
            if excluded_titles:
                params.extend(excluded_titles)
            df = pd.read_sql_query(query.format(excluded_clause=excluded_clause, gender_filter=gf), conn, params=params)
            self.logger.info(f"Pobrano {len(df)} postów dla {forum_name} - {gender}")
            return df
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania danych: {e}")
            raise

    def get_forum_data_multi(self, conn: sqlite3.Connection, forum_names: List[str], gender: str) -> pd.DataFrame:
        """Pobiera dane łącznie dla wielu forów i konkretnej płci"""
        self.logger.info(f"Pobieranie danych dla forów: {', '.join(forum_names)}, płeć: {gender}")
        placeholders = ",".join(["?"] * len(forum_names))
        # Przygotuj filtr sekcji (case-insensitive)
        excluded_titles = [t.lower() for t in (EXCLUDED_SECTIONS or [])]
        excluded_clause = ""
        if excluded_titles:
            _ph = ",".join(["?"] * len(excluded_titles))
            excluded_clause = f" AND LOWER(fs.title) NOT IN ({_ph})"
        gender_all = str(gender).upper() == 'KM'
        if self._gender_predictions_available(conn):
            query = f"""
            SELECT 
                fp.id,
                fp.user_id AS user_id,
                fp.content,
                fp.post_date,
                fp.username,
                fu.gender AS db_gender,
                COALESCE(fu.gender, gp.predicted_gender) AS eff_gender,
                fu.religion,
                fu.localization,
                ft.title as thread_title,
                f.spider_name,
                ta.token_count AS token_count
            FROM forum_posts fp
            JOIN forum_threads ft ON fp.thread_id = ft.id
            JOIN forum_sections fs ON ft.section_id = fs.id
            JOIN forums f ON fs.forum_id = f.id
            LEFT JOIN forum_users fu ON fp.user_id = fu.id
            LEFT JOIN (
                SELECT gp1.user_id, gp1.predicted_gender
                FROM gender_predictions gp1
                JOIN (
                    SELECT user_id, MAX(COALESCE(updated_at, created_at)) AS ts
                    FROM gender_predictions
                    GROUP BY user_id
                ) last ON last.user_id = gp1.user_id AND COALESCE(gp1.updated_at, gp1.created_at) = last.ts
            ) gp ON gp.user_id = fu.id
            JOIN token_analysis ta ON ta.post_id = fp.id
            WHERE f.spider_name IN ({placeholders})
              {{gender_filter}}
              AND ta.token_count BETWEEN ? AND ?
              {excluded_clause}
              AND fp.content IS NOT NULL
              AND LENGTH(fp.content) > 0
            """
        else:
            query = f"""
            SELECT 
                fp.id,
                fp.user_id AS user_id,
                fp.content,
                fp.post_date,
                fp.username,
                fu.gender AS db_gender,
                fu.gender AS eff_gender,
                fu.religion,
                fu.localization,
                ft.title as thread_title,
                f.spider_name,
                ta.token_count AS token_count
            FROM forum_posts fp
            JOIN forum_threads ft ON fp.thread_id = ft.id
            JOIN forum_sections fs ON ft.section_id = fs.id
            JOIN forums f ON fs.forum_id = f.id
            LEFT JOIN forum_users fu ON fp.user_id = fu.id
            JOIN token_analysis ta ON ta.post_id = fp.id
            WHERE f.spider_name IN ({placeholders})
              {{gender_filter}}
              AND ta.token_count BETWEEN ? AND ?
              {excluded_clause}
              AND fp.content IS NOT NULL
              AND LENGTH(fp.content) > 0
            """
        try:
            if gender_all:
                gf = ""
                params = list(forum_names) + [self.min_tokens, self.max_tokens]
            else:
                gf = "AND COALESCE(fu.gender, gp.predicted_gender) = ?" if self._gender_predictions_available(conn) else "AND fu.gender = ?"
                params = list(forum_names) + [gender, self.min_tokens, self.max_tokens]
            if excluded_titles:
                params.extend(excluded_titles)
            df = pd.read_sql_query(query.format(excluded_clause=excluded_clause, gender_filter=gf), conn, params=params)
            self.logger.info(f"Pobrano {len(df)} postów łącznie dla forów ({', '.join(forum_names)}) - {gender}")
            return df
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania danych łączonych: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Przetwarza tekst przed analizą"""
        if not isinstance(text, str):
            return ""
        
        # Usuwa HTML tagi
        text = re.sub(r'<[^>]+>', '', text)
        
        # Usuwa linki
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Usuwa znaki specjalne, zostawia polskie znaki
        text = re.sub(r'[^\w\sąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', ' ', text)
        
        # Usuwa nadmiarowe spacje
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def filter_posts_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtruje posty według długości (liczba tokenów)"""
        self.logger.info("Filtrowanie postów według długości...")
        # Jeśli kolumna token_count pochodzi z bazy, użyj jej; w przeciwnym razie oszacuj
        if 'token_count' in df.columns:
            filtered_df = df[(df['token_count'] >= self.min_tokens) & (df['token_count'] <= self.max_tokens)].copy()
        else:
            df['token_count'] = df['content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
            filtered_df = df[(df['token_count'] >= self.min_tokens) & (df['token_count'] <= self.max_tokens)].copy()
        self.logger.info(f"Po filtrowaniu długości: {len(filtered_df)} postów")
        return filtered_df
    
    def limit_posts_per_user(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ogranicza liczbę postów na użytkownika"""
        self.logger.info("Ograniczanie liczby postów na użytkownika...")
        
        # Grupowanie po użytkowniku i losowanie postów
        limited_posts = []
        
        group_key = 'user_id' if 'user_id' in df.columns else 'username'
        for user_key, group in df.groupby(group_key):
            if len(group) > self.max_posts_per_user:
                # Losowo wybierz maksymalną liczbę postów
                selected_posts = group.sample(n=self.max_posts_per_user, random_state=self.random_seed)
                limited_posts.append(selected_posts)
            else:
                limited_posts.append(group)
        
        result_df = pd.concat(limited_posts, ignore_index=True)
        self.logger.info(f"Po ograniczeniu postów na użytkownika: {len(result_df)} postów")
        return result_df
    
    def balance_gender_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balansuje dane między płciami"""
        self.logger.info("Balansowanie danych między płciami...")
        
        # Sprawdź dostępne płcie
        gender_counts = df['gender'].value_counts()
        self.logger.info(f"Liczba postów według płci:\n{gender_counts}")
        
        if len(gender_counts) < 2:
            self.logger.warning("Brak danych dla obu płci")
            return df
        
        # Znajdź mniejszą liczbę postów
        min_count = gender_counts.min()
        
        # Zbalansuj dane
        balanced_posts = []
        for gender in gender_counts.index:
            gender_posts = df[df['gender'] == gender]
            if len(gender_posts) > min_count:
                # Losowo wybierz posty
                selected_posts = gender_posts.sample(n=min_count, random_state=42)
                balanced_posts.append(selected_posts)
            else:
                balanced_posts.append(gender_posts)
        
        result_df = pd.concat(balanced_posts, ignore_index=True)
        self.logger.info(f"Po balansowaniu: {len(result_df)} postów")
        return result_df
    
    def prepare_data_for_top2vec(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Przygotowuje dane dla Top2Vec"""
        self.logger.info("Przygotowywanie danych dla Top2Vec...")
        
        # Przetwarzanie tekstu
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Usuń puste posty
        df = df[df['processed_content'].str.len() > 0]
        
        # Konwertuj na listę dokumentów i oryginalnych ID postów
        documents = df['processed_content'].tolist()
        document_ids = df['id'].astype(str).tolist()
        
        self.logger.info(f"Przygotowano {len(documents)} dokumentów dla analizy")
        return documents, document_ids
    
    def train_top2vec_model(self, documents: List[str], forum_name: str, gender: str, document_ids: Optional[List[str]] = None) -> Top2Vec:
        """Trenuje model Top2Vec"""
        self.logger.info(f"Trenowanie modelu Top2Vec dla {forum_name} - {gender}")
        
        try:
            # Trenowanie modelu
            model = Top2Vec(
                documents=documents,
                embedding_model=self.embedding_model,
                min_count=5,  # Minimalna liczba wystąpień słowa
                document_ids=document_ids,
                keep_documents=self.keep_documents_in_model,
                verbose=True
            )
            
            self.logger.info(f"Model wytrenowany. Liczba tematów: {model.get_num_topics()}")

            # Opcjonalna redukcja do docelowej liczby tematów (obsługa różnych API)
            model = self._reduce_to_target_topics(model)
            return model
            
        except Exception as e:
            self.logger.error(f"Błąd podczas trenowania modelu: {e}")
            raise
    
    def save_model(self, model: Top2Vec, forum_name: str | List[str], gender: str):
        """Zapisuje wytrenowany model"""
        comps = self._get_run_components(forum_name, gender)
        model_path = comps['models_dir'] / "model"
        
        try:
            # Upewnij się, że katalog docelowy istnieje
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            self.logger.info(f"Model zapisany w: {model_path}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania modelu: {e}")
            raise
    
    def analyze_topics(self, model: Top2Vec, forum_name: str, gender: str, total_documents: Optional[int] = None, df_for_stats: Optional[pd.DataFrame] = None, input_params: Optional[Dict] = None) -> Dict:
        """Analizuje tematy z modelu"""
        self.logger.info("Analizowanie tematów...")
        
        # Jeśli proszono o target_num_topics i wersja Top2Vec wspiera reduced=True,
        # preferuj zredukowaną przestrzeń tematów przy raportowaniu.
        prefer_reduced = isinstance(self.target_num_topics, int) and self.target_num_topics >= 2
        try:
            num_topics = self._call_with_reduced(model.get_num_topics, prefer_reduced=prefer_reduced)
        except Exception:
            num_topics = model.get_num_topics()
        self.logger.info(f"Analizowanie {num_topics} tematów")
        
        run_slug = self.run_slug or self._build_run_slug(forum_name, gender)
        # Oszacuj łączną liczbę dokumentów, jeśli nie podano
        if total_documents is None:
            try:
                sizes_sum = int(np.sum(model.get_topic_sizes()[0]))
                total_documents = sizes_sum
            except Exception:
                total_documents = None

        # Statystyki zbioru
        overall_stats = None
        per_forum_stats = None
        if df_for_stats is not None and len(df_for_stats) > 0:
            overall_stats = {
                'posts': int(len(df_for_stats)),
                'unique_users': int(df_for_stats['user_id'].nunique() if 'user_id' in df_for_stats.columns else df_for_stats['username'].nunique()),
                'total_tokens': int(df_for_stats['token_count'].sum() if 'token_count' in df_for_stats.columns else 0)
            }
            try:
                per_forum = []
                for forum, g in df_for_stats.groupby('spider_name'):
                    per_forum.append({
                        'forum': forum,
                        'posts': int(len(g)),
                        'unique_users': int(g['user_id'].nunique() if 'user_id' in g.columns else g['username'].nunique()),
                        'total_tokens': int(g['token_count'].sum() if 'token_count' in g.columns else 0)
                    })
                per_forum_stats = per_forum
            except Exception:
                per_forum_stats = None

        analysis_results = {
            'forum_name': forum_name,
            'gender': gender,
            'num_topics': num_topics,
            'total_documents': int(total_documents) if isinstance(total_documents, (int, np.integer)) else total_documents,
            'dataset_stats': {
                'overall': overall_stats,
                'per_forum': per_forum_stats,
            },
            'input_params': input_params or {},
            'topics': [],
            'topic_hierarchy': None,
            'timestamp': datetime.now().isoformat(),
            'run_slug': run_slug,
            'embedding_model': self.embedding_model
        }
        
        # Mapowanie: topic_num -> rozmiar
        topic_sizes_map = {}
        try:
            try:
                sizes, size_topic_nums = self._call_with_reduced(model.get_topic_sizes, prefer_reduced=prefer_reduced)
            except Exception:
                sizes, size_topic_nums = model.get_topic_sizes()
            topic_sizes_map = {int(tn): int(sz) for sz, tn in zip(sizes, size_topic_nums)}
        except Exception as e:
            self.logger.warning(f"Nie udało się pobrać rozmiarów tematów: {e}")
        
        # Pobierz słowa i wyniki dla wszystkich tematów zgodnie z API Top2Vec
        try:
            try:
                got = self._call_with_reduced(model.get_topics, prefer_reduced=prefer_reduced)
            except Exception:
                got = model.get_topics()
            # wspieraj 3 lub 4 elementy zwracane przez różne wersje
            if isinstance(got, tuple) and len(got) == 4:
                topic_words_all, word_scores_all, topic_scores_all, topic_nums = got
            elif isinstance(got, tuple) and len(got) == 3:
                topic_words_all, word_scores_all, topic_nums = got
                topic_scores_all = None
            else:
                # Nieoczekiwany format – spróbuj starego API bez reduced
                topic_words_all, word_scores_all, topic_nums = model.get_topics()
                topic_scores_all = None
        except Exception:
            # Fallback do pobierania tematów jeden po drugim
            topic_nums = list(range(num_topics))
            topic_words_all, word_scores_all, topic_scores_all = [], [], None
            for tnum in topic_nums:
                try:
                    words, scores = model.get_topic_words(tnum)
                    topic_words_all.append(words)
                    word_scores_all.append(scores)
                except Exception as e:
                    self.logger.warning(f"Nie udało się pobrać słów tematu {tnum}: {e}")
                    topic_words_all.append([])
                    word_scores_all.append([])
        
        # Zbierz metryki dla tematów (od biblioteki)
        for idx, tnum in enumerate(topic_nums):
            try:
                words = topic_words_all[idx]
                scores = word_scores_all[idx]
                words_list = (words.tolist() if hasattr(words, 'tolist') else list(words))[:20]
                scores_list = (scores.tolist() if hasattr(scores, 'tolist') else list(scores))[:20]
                num_docs_in_topic = topic_sizes_map.get(int(tnum), 0)
                avg_doc_score = 0.0
                top_doc_ids: List[str] = []
                try:
                    docs, doc_scores, doc_ids = model.search_documents_by_topic(int(tnum), num_docs=min(200, max(20, num_docs_in_topic)))
                    if doc_scores is not None and len(doc_scores) > 0:
                        avg_doc_score = float(np.mean(doc_scores))
                    if doc_ids is not None:
                        top_doc_ids = [str(d) for d in list(doc_ids)[:10]]
                except Exception as e:
                    self.logger.debug(f"search_documents_by_topic nie powiodło się dla tematu {tnum}: {e}")
                topic_info = {
                    'topic_id': int(tnum),
                    'words': words_list,
                    'word_scores': scores_list,
                    'num_documents': int(num_docs_in_topic),
                    'avg_doc_score': avg_doc_score,
                    'top_document_ids': top_doc_ids
                }
                analysis_results['topics'].append(topic_info)
            except Exception as e:
                self.logger.warning(f"Błąd podczas analizy tematu {tnum}: {e}")
                continue
        
        # Hierarchia tematów
        try:
            topic_h = None
            # Najpierw spróbuj natywnego gettera (jeśli już policzona)
            if hasattr(model, 'get_topic_hierarchy'):
                try:
                    topic_h = model.get_topic_hierarchy()
                except Exception:
                    topic_h = None

            # Jeśli brak, spróbuj wywołać redukcję hierarchiczną do sensownej liczby tematów
            if topic_h is None and hasattr(model, 'hierarchical_topic_reduction'):
                try:
                    target = max(2, min(50, num_topics // 2 or 2))
                    reduced = model.hierarchical_topic_reduction(target)
                    # Niektóre implementacje zwracają strukturę hierarchii bezpośrednio
                    if reduced is not None:
                        topic_h = reduced
                except Exception:
                    topic_h = None
                # Po redukcji spróbuj ponownie pobrać hierarchię, jeśli getter istnieje
                if topic_h is None and hasattr(model, 'get_topic_hierarchy'):
                    try:
                        topic_h = model.get_topic_hierarchy()
                    except Exception:
                        topic_h = None

            if topic_h is not None:
                analysis_results['topic_hierarchy'] = topic_h.tolist() if hasattr(topic_h, 'tolist') else topic_h
            else:
                self.logger.info("Hierarchia tematów niedostępna w tej wersji; pomijam.")
        except Exception as e:
            self.logger.warning(f"Nie udało się pobrać hierarchii tematów: {e}")
        
        return analysis_results
    
    def generate_visualizations(self, model: Top2Vec, analysis_results: Dict, forum_name: str | List[str], gender: str):
        """Generuje wizualizacje tematów"""
        self.logger.info("Generowanie wizualizacji...")
        comps = self._get_run_components(forum_name, gender)
        results_dir = comps['results_dir']
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Chmura słów dla każdego tematu
        self._generate_topic_wordclouds(model, analysis_results, results_dir, analysis_results.get('run_slug', self.run_slug or 'run'))
        
        # 2. Wykres słupkowy liczby dokumentów w tematach
        self._generate_topic_distribution_chart(analysis_results, results_dir, analysis_results.get('run_slug', self.run_slug or 'run'))
        
        # 3. Wykres słupkowy średnich wyników dokumentów
        self._generate_doc_scores_chart(analysis_results, results_dir, analysis_results.get('run_slug', self.run_slug or 'run'))
        
        # 4. Interaktywna mapa tematów (jeśli dostępna)
        self._generate_topic_map(model, analysis_results, results_dir, analysis_results.get('run_slug', self.run_slug or 'run'))
    
    def _generate_topic_wordclouds(self, model: Top2Vec, analysis_results: Dict, results_dir: Path, run_slug: str):
        """Generuje chmury słów dla tematów"""
        try:
            wc_dir = results_dir / "wordcloud"
            wc_dir.mkdir(parents=True, exist_ok=True)
            for topic_info in analysis_results['topics']:
                topic_id = topic_info['topic_id']
                words = topic_info['words']
                scores = topic_info['word_scores']
                
                # Tworzenie słownika słowo -> waga
                word_weights = dict(zip(words, scores))
                
                # Generowanie chmury słów
                wordcloud = WordCloud(
                    width=800, height=600,
                    background_color='white',
                    max_words=50,
                    colormap='viridis'
                ).generate_from_frequencies(word_weights)
                
                # Zapisanie
                plt.figure(figsize=(10, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Temat {topic_id} - {analysis_results.get("run_slug", run_slug)}')
                
                wordcloud_path = wc_dir / f"topic_{topic_id}_wordcloud.png"
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"Błąd podczas generowania chmur słów: {e}")
    
    def _generate_topic_distribution_chart(self, analysis_results: Dict, results_dir: Path, run_slug: str):
        """Generuje wykres rozkładu dokumentów w tematach"""
        try:
            topics = [f"Temat {t['topic_id']}" for t in analysis_results['topics']]
            doc_counts = [t['num_documents'] for t in analysis_results['topics']]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(topics)), doc_counts, color='skyblue', alpha=0.7)
            
            plt.xlabel('Tematy')
            plt.ylabel('Liczba dokumentów')
            plt.title(f'Rozkład dokumentów w tematach - {analysis_results.get("run_slug", run_slug)}')
            plt.xticks(range(len(topics)), topics, rotation=45, ha='right')
            
            # Dodanie wartości na słupkach
            for bar, count in zip(bars, doc_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(doc_counts),
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            chart_path = results_dir / "topic_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Błąd podczas generowania wykresu rozkładu: {e}")
    
    def _generate_doc_scores_chart(self, analysis_results: Dict, results_dir: Path, run_slug: str):
        """Generuje wykres średnich wyników dokumentów"""
        try:
            topics = [f"Temat {t['topic_id']}" for t in analysis_results['topics']]
            avg_scores = [t['avg_doc_score'] for t in analysis_results['topics']]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(topics)), avg_scores, color='lightcoral', alpha=0.7)
            
            plt.xlabel('Tematy')
            plt.ylabel('Średni wynik dokumentu')
            plt.title(f'Średnie wyniki dokumentów w tematach - {analysis_results.get("run_slug", run_slug)}')
            plt.xticks(range(len(topics)), topics, rotation=45, ha='right')
            
            # Dodanie wartości na słupkach
            for bar, score in zip(bars, avg_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001*max(avg_scores),
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            chart_path = results_dir / "doc_scores_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Błąd podczas generowania wykresu wyników: {e}")
    
    def _generate_topic_map(self, model: Top2Vec, analysis_results: Dict, results_dir: Path, run_slug: str):
        """Generuje interaktywną mapę tematów.
        Fallback: jeśli brak get_topic_coordinates(), redukujemy wektory tematów do 2D metodą PCA.
        """
        try:
            coords = None
            # Próba użycia metody (jeśli dostępna)
            try:
                topic_coords = model.get_topic_coordinates()  # może nie istnieć w tej wersji
                if topic_coords is not None:
                    coords = topic_coords
            except Exception:
                coords = None

            # Fallback: użyj wektorów tematów i zredukuj do 2D PCA (bez dodatkowych zależności)
            if coords is None:
                topic_vectors = getattr(model, 'topic_vectors', None)
                if topic_vectors is None:
                    raise RuntimeError("Brak get_topic_coordinates oraz topic_vectors")
                X = np.array(topic_vectors)
                if X.ndim != 2 or X.shape[0] < 2:
                    raise RuntimeError("Za mało tematów do rzutowania 2D")
                Xc = X - X.mean(axis=0, keepdims=True)
                # PCA przez SVD
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                coords = U[:, :2] * S[:2]

            # Tworzenie interaktywnej mapy
            fig = px.scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                text=[f"Temat {i}" for i in range(coords.shape[0])],
                title=f'Mapa tematów - {analysis_results.get("run_slug", run_slug)}'
            )
            fig.update_traces(
                textposition="top center",
                marker=dict(size=15, color='red', opacity=0.7)
            )
            fig.update_layout(
                xaxis_title="Wymiar 1",
                yaxis_title="Wymiar 2",
                showlegend=False
            )
            # Zapisanie jako HTML
            map_path = results_dir / "topic_map.html"
            fig.write_html(str(map_path))
        except Exception as e:
            self.logger.warning(f"Błąd podczas generowania mapy tematów: {e}")
    
    def save_analysis_results(self, analysis_results: Dict, forum_name: str | List[str], gender: str):
        """Zapisuje wyniki analizy"""
        comps = self._get_run_components(forum_name, gender)
        results_dir = comps['results_dir']
        
        # Zapisanie jako JSON
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Wyniki analizy zapisane w: {json_path}")
            # Skopiuj bieżący plik konfiguracji pipeline (jeśli ścieżka dostępna)
            cfg_path = os.environ.get('TOPIC_PIPELINE_CONFIG_PATH')
            try:
                if cfg_path and Path(cfg_path).exists():
                    shutil.copy2(cfg_path, results_dir / "pipeline.config.json")
            except Exception as e:
                self.logger.warning(f"Nie udało się skopiować pipeline.config.json: {e}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania wyników: {e}")
            raise
    
    def run_analysis(self, forum_name: str, gender: str):
        """Uruchamia pełną analizę dla konkretnego forum i płci"""
        # Zbuduj slug i logger z nową strukturą katalogów
        self.current_run_time = datetime.now()
        self.run_slug = self._build_run_slug(forum_name, gender, when=self.current_run_time)
        comps = self._get_run_components(forum_name, gender)
        Path(comps['logs_dir']).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(Path(comps['logs_dir']), self.run_slug)
        
        self.logger.info(f"Rozpoczęcie analizy dla {forum_name} - {gender}")
        
        try:
            # Połączenie z bazą danych

            conn = self.connect_to_db()
            
            # Pobranie danych
            df = self.get_forum_data(conn, forum_name, gender)
            
            if len(df) == 0:
                self.logger.warning(f"Brak danych dla {forum_name} - {gender}")
                return
            
            # Filtrowanie i przygotowanie danych
            df = self.filter_posts_by_length(df)
            df = self.limit_posts_per_user(df)
            
            # Przygotowanie danych dla Top2Vec
            documents, document_ids = self.prepare_data_for_top2vec(df)
            
            if len(documents) < 10:
                self.logger.warning(f"Za mało dokumentów ({len(documents)}) do analizy")
                return
            
            # Trenowanie modelu
            model = self.train_top2vec_model(documents, forum_name, gender, document_ids=document_ids)
            
            # Analiza tematów
            params = self._collect_run_params([forum_name], gender, is_combined=False)
            analysis_results = self.analyze_topics(model, forum_name, gender, total_documents=len(documents), df_for_stats=df, input_params=params)
            
            # Generowanie wizualizacji
            self.generate_visualizations(model, analysis_results, forum_name, gender)
            
            # Zapisanie wyników
            self.save_analysis_results(analysis_results, forum_name, gender)
            
            # Zapisanie modelu
            self.save_model(model, forum_name, gender)
            
            self.logger.info(f"Analiza zakończona pomyślnie dla {forum_name} - {gender}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas analizy: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def run_combined_analysis(self, forum_names: List[str], gender: str):
        """Uruchamia pełną analizę dla łączonych forów (jeden model per płeć)"""
        combined_name = ",".join(forum_names)
        # Slug dla łączonych: kody połączone podkreślnikami
        self.current_run_time = datetime.now()
        self.run_slug = self._build_run_slug(forum_names, gender, when=self.current_run_time)
        comps = self._get_run_components(forum_names, gender)
        Path(comps['logs_dir']).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(Path(comps['logs_dir']), self.run_slug)
        self.logger.info(f"Rozpoczęcie analizy łączonej dla {combined_name} - {gender}")
        try:
            conn = self.connect_to_db()
            df = self.get_forum_data_multi(conn, forum_names, gender)
            if len(df) == 0:
                self.logger.warning(f"Brak danych dla {combined_name} - {gender}")
                return
            df = self.filter_posts_by_length(df)
            df = self.limit_posts_per_user(df)
            # Uwaga: balansowanie płci nie ma sensu w modelu 1-płciowym; pomijamy
            documents, document_ids = self.prepare_data_for_top2vec(df)
            if len(documents) < 10:
                self.logger.warning(f"Za mało dokumentów ({len(documents)}) do analizy łączonej")
                return
            model = self.train_top2vec_model(documents, combined_name, gender, document_ids=document_ids)
            params = self._collect_run_params(forum_names, gender, is_combined=True)
            analysis_results = self.analyze_topics(model, combined_name, gender, total_documents=len(documents), df_for_stats=df, input_params=params)
            self.generate_visualizations(model, analysis_results, forum_names, gender)
            self.save_analysis_results(analysis_results, forum_names, gender)
            self.save_model(model, forum_names, gender)
            self.logger.info(f"Analiza łączona zakończona: {combined_name} - {gender}")
        except Exception as e:
            self.logger.error(f"Błąd podczas analizy łączonej: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

def main():
    """Główna funkcja"""
    # Konfiguracja
    db_path = "data/databases/analysis_forums.db"
    output_dir = Path("data/topics")
    
    # Sprawdzenie czy baza istnieje
    if not Path(db_path).exists():
        print(f"Baza danych nie istnieje: {db_path}")
        return
    
    # Inicjalizacja analizatora
    analyzer = TopicModelingAnalyzer(db_path, output_dir)
    
    # Fora do analizy
    forums = ['radio_katolik', 'dolina_modlitwy']
    genders = ['M', 'K']  # Poprawione: M (male), K (female) zgodnie z bazą danych
    
    # Uruchomienie analizy dla każdej kombinacji
    for forum in forums:
        for gender in genders:
            try:
                print(f"\n{'='*60}")
                print(f"Analiza: {forum} - {gender}")
                print(f"{'='*60}")
                
                analyzer.run_analysis(forum, gender)
                
                print(f"✓ Analiza zakończona: {forum} - {gender}")
                
            except Exception as e:
                print(f"✗ Błąd podczas analizy {forum} - {gender}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("Wszystkie analizy zakończone!")
    print(f"Wyniki zapisane w: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

