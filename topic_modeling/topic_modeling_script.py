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
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Top2Vec i biblioteki do analizy
from top2vec import Top2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Konfiguracja logowania
def setup_logging(log_dir: Path, forum_name: str, gender: str) -> logging.Logger:
    """Konfiguruje logowanie dla konkretnego forum i płci"""
    log_file = log_dir / f"{forum_name}_{gender}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(f"{forum_name}_{gender}")

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
        
    def connect_to_db(self) -> sqlite3.Connection:
        """Nawiązuje połączenie z bazą danych"""
        try:
            conn = sqlite3.connect(self.db_path)
            self.logger.info(f"Połączono z bazą danych: {self.db_path}")
            return conn
        except Exception as e:
            self.logger.error(f"Błąd połączenia z bazą danych: {e}")
            raise
    
    def get_forum_data(self, conn: sqlite3.Connection, forum_name: str, gender: str) -> pd.DataFrame:
        """Pobiera dane z konkretnego forum i płci"""
        self.logger.info(f"Pobieranie danych dla forum: {forum_name}, płeć: {gender}")
        
        query = """
        SELECT 
            fp.id,
            fp.content,
            fp.post_date,
            fp.username,
            fu.gender,
            fu.religion,
            fu.localization,
            ft.title as thread_title,
            f.spider_name
        FROM forum_posts fp
        JOIN forum_threads ft ON fp.thread_id = ft.id
        JOIN forum_sections fs ON ft.section_id = fs.id
        JOIN forums f ON fs.forum_id = f.id
        LEFT JOIN forum_users fu ON fp.user_id = fu.id
        WHERE f.spider_name = ? 
        AND fu.gender = ?
        AND fp.content IS NOT NULL
        AND LENGTH(fp.content) > 0
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=(forum_name, gender))
            self.logger.info(f"Pobrano {len(df)} postów dla {forum_name} - {gender}")
            return df
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania danych: {e}")
            raise

    def get_forum_data_multi(self, conn: sqlite3.Connection, forum_names: List[str], gender: str) -> pd.DataFrame:
        """Pobiera dane łącznie dla wielu forów i konkretnej płci"""
        self.logger.info(f"Pobieranie danych dla forów: {', '.join(forum_names)}, płeć: {gender}")
        placeholders = ",".join(["?"] * len(forum_names))
        query = f"""
        SELECT 
            fp.id,
            fp.content,
            fp.post_date,
            fp.username,
            fu.gender,
            fu.religion,
            fu.localization,
            ft.title as thread_title,
            f.spider_name
        FROM forum_posts fp
        JOIN forum_threads ft ON fp.thread_id = ft.id
        JOIN forum_sections fs ON ft.section_id = fs.id
        JOIN forums f ON fs.forum_id = f.id
        LEFT JOIN forum_users fu ON fp.user_id = fu.id
        WHERE f.spider_name IN ({placeholders})
        AND fu.gender = ?
        AND fp.content IS NOT NULL
        AND LENGTH(fp.content) > 0
        """
        try:
            params = list(forum_names) + [gender]
            df = pd.read_sql_query(query, conn, params=params)
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
        
        # Szacowanie liczby tokenów (słowa)
        df['token_count'] = df['content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Filtrowanie według długości
        filtered_df = df[
            (df['token_count'] >= self.min_tokens) & 
            (df['token_count'] <= self.max_tokens)
        ].copy()
        
        self.logger.info(f"Po filtrowaniu długości: {len(filtered_df)} postów")
        return filtered_df
    
    def limit_posts_per_user(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ogranicza liczbę postów na użytkownika"""
        self.logger.info("Ograniczanie liczby postów na użytkownika...")
        
        # Grupowanie po użytkowniku i losowanie postów
        limited_posts = []
        
        for username, group in df.groupby('username'):
            if len(group) > self.max_posts_per_user:
                # Losowo wybierz maksymalną liczbę postów
                selected_posts = group.sample(n=self.max_posts_per_user, random_state=42)
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
    
    def prepare_data_for_top2vec(self, df: pd.DataFrame) -> List[str]:
        """Przygotowuje dane dla Top2Vec"""
        self.logger.info("Przygotowywanie danych dla Top2Vec...")
        
        # Przetwarzanie tekstu
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Usuń puste posty
        df = df[df['processed_content'].str.len() > 0]
        
        # Konwertuj na listę
        documents = df['processed_content'].tolist()
        
        self.logger.info(f"Przygotowano {len(documents)} dokumentów dla analizy")
        return documents
    
    def train_top2vec_model(self, documents: List[str], forum_name: str, gender: str) -> Top2Vec:
        """Trenuje model Top2Vec"""
        self.logger.info(f"Trenowanie modelu Top2Vec dla {forum_name} - {gender}")
        
        try:
            # Trenowanie modelu
            model = Top2Vec(
                documents=documents,
                min_count=5,  # Minimalna liczba wystąpień słowa
                topic_merge_delta=0.1,  # Parametr łączenia tematów
                topic_merge_threshold=0.1,  # Próg łączenia tematów
                verbose=True
            )
            
            self.logger.info(f"Model wytrenowany. Liczba tematów: {model.get_num_topics()}")
            return model
            
        except Exception as e:
            self.logger.error(f"Błąd podczas trenowania modelu: {e}")
            raise
    
    def save_model(self, model: Top2Vec, forum_name: str, gender: str):
        """Zapisuje wytrenowany model"""
        model_path = self.output_dir / "models" / forum_name / gender / f"{forum_name}_{gender}_model"
        
        try:
            # Upewnij się, że katalog docelowy istnieje
            (self.output_dir / "models" / forum_name / gender).mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            self.logger.info(f"Model zapisany w: {model_path}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania modelu: {e}")
            raise
    
    def analyze_topics(self, model: Top2Vec, forum_name: str, gender: str) -> Dict:
        """Analizuje tematy z modelu"""
        self.logger.info("Analizowanie tematów...")
        
        num_topics = model.get_num_topics()
        self.logger.info(f"Analizowanie {num_topics} tematów")
        
        analysis_results = {
            'forum_name': forum_name,
            'gender': gender,
            'num_topics': num_topics,
            'topics': [],
            'topic_hierarchy': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Analiza każdego tematu
        for topic_id in range(num_topics):
            try:
                # Pobierz słowa kluczowe tematu
                topic_words, word_scores = model.get_topic(topic_id)
                
                # Pobierz dokumenty dla tematu
                topic_docs, doc_scores = model.get_documents(topic_id)
                
                topic_info = {
                    'topic_id': topic_id,
                    'words': topic_words[:20].tolist(),  # Top 20 słów
                    'word_scores': word_scores[:20].tolist(),
                    'num_documents': len(topic_docs),
                    'avg_doc_score': float(np.mean(doc_scores)) if len(doc_scores) > 0 else 0.0
                }
                
                analysis_results['topics'].append(topic_info)
                
            except Exception as e:
                self.logger.warning(f"Błąd podczas analizy tematu {topic_id}: {e}")
                continue
        
        # Hierarchia tematów
        try:
            topic_hierarchy = model.get_topic_hierarchy()
            analysis_results['topic_hierarchy'] = topic_hierarchy.tolist()
        except Exception as e:
            self.logger.warning(f"Nie udało się pobrać hierarchii tematów: {e}")
        
        return analysis_results
    
    def generate_visualizations(self, model: Top2Vec, analysis_results: Dict, forum_name: str, gender: str):
        """Generuje wizualizacje tematów"""
        self.logger.info("Generowanie wizualizacji...")
        
        results_dir = self.output_dir / "results" / forum_name / gender
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Chmura słów dla każdego tematu
        self._generate_topic_wordclouds(model, analysis_results, results_dir)
        
        # 2. Wykres słupkowy liczby dokumentów w tematach
        self._generate_topic_distribution_chart(analysis_results, results_dir)
        
        # 3. Wykres słupkowy średnich wyników dokumentów
        self._generate_doc_scores_chart(analysis_results, results_dir)
        
        # 4. Interaktywna mapa tematów (jeśli dostępna)
        self._generate_topic_map(model, analysis_results, results_dir)
    
    def _generate_topic_wordclouds(self, model: Top2Vec, analysis_results: Dict, results_dir: Path):
        """Generuje chmury słów dla tematów"""
        try:
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
                plt.title(f'Temat {topic_id} - {analysis_results["forum_name"]} ({analysis_results["gender"]})')
                
                wordcloud_path = results_dir / f"topic_{topic_id}_wordcloud.png"
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"Błąd podczas generowania chmur słów: {e}")
    
    def _generate_topic_distribution_chart(self, analysis_results: Dict, results_dir: Path):
        """Generuje wykres rozkładu dokumentów w tematach"""
        try:
            topics = [f"Temat {t['topic_id']}" for t in analysis_results['topics']]
            doc_counts = [t['num_documents'] for t in analysis_results['topics']]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(topics)), doc_counts, color='skyblue', alpha=0.7)
            
            plt.xlabel('Tematy')
            plt.ylabel('Liczba dokumentów')
            plt.title(f'Rozkład dokumentów w tematach - {analysis_results["forum_name"]} ({analysis_results["gender"]})')
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
    
    def _generate_doc_scores_chart(self, analysis_results: Dict, results_dir: Path):
        """Generuje wykres średnich wyników dokumentów"""
        try:
            topics = [f"Temat {t['topic_id']}" for t in analysis_results['topics']]
            avg_scores = [t['avg_doc_score'] for t in analysis_results['topics']]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(topics)), avg_scores, color='lightcoral', alpha=0.7)
            
            plt.xlabel('Tematy')
            plt.ylabel('Średni wynik dokumentu')
            plt.title(f'Średnie wyniki dokumentów w tematach - {analysis_results["forum_name"]} ({analysis_results["gender"]})')
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
    
    def _generate_topic_map(self, model: Top2Vec, analysis_results: Dict, results_dir: Path):
        """Generuje interaktywną mapę tematów"""
        try:
            # Pobierz współrzędne tematów
            topic_coords = model.get_topic_coordinates()
            
            if topic_coords is not None:
                # Tworzenie interaktywnej mapy
                fig = px.scatter(
                    x=topic_coords[:, 0],
                    y=topic_coords[:, 1],
                    text=[f"Temat {i}" for i in range(len(topic_coords))],
                    title=f'Mapa tematów - {analysis_results["forum_name"]} ({analysis_results["gender"]})'
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
    
    def save_analysis_results(self, analysis_results: Dict, forum_name: str, gender: str):
        """Zapisuje wyniki analizy"""
        results_dir = self.output_dir / "results" / forum_name / gender
        
        # Zapisanie jako JSON
        json_path = results_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Wyniki analizy zapisane w: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania wyników: {e}")
            raise
    
    def run_analysis(self, forum_name: str, gender: str):
        """Uruchamia pełną analizę dla konkretnego forum i płci"""
        self.logger = setup_logging(
            self.output_dir / "logs", 
            forum_name, 
            gender
        )
        
        self.logger.info(f"Rozpoczęcie analizy dla {forum_name} - {gender}")
        
        try:
            # Połączenie z bazą danych
            # Utwórz katalogi wyjściowe, jeśli nie istnieją
            (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "results").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

            conn = self.connect_to_db()
            
            # Pobranie danych
            df = self.get_forum_data(conn, forum_name, gender)
            
            if len(df) == 0:
                self.logger.warning(f"Brak danych dla {forum_name} - {gender}")
                return
            
            # Filtrowanie i przygotowanie danych
            df = self.filter_posts_by_length(df)
            df = self.limit_posts_per_user(df)
            df = self.balance_gender_data(df)
            
            # Przygotowanie danych dla Top2Vec
            documents = self.prepare_data_for_top2vec(df)
            
            if len(documents) < 10:
                self.logger.warning(f"Za mało dokumentów ({len(documents)}) do analizy")
                return
            
            # Trenowanie modelu
            model = self.train_top2vec_model(documents, forum_name, gender)
            
            # Analiza tematów
            analysis_results = self.analyze_topics(model, forum_name, gender)
            
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
        combined_name = "+".join(forum_names)
        self.logger = setup_logging(
            self.output_dir / "logs",
            combined_name,
            gender
        )
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
            documents = self.prepare_data_for_top2vec(df)
            if len(documents) < 10:
                self.logger.warning(f"Za mało dokumentów ({len(documents)}) do analizy łączonej")
                return
            model = self.train_top2vec_model(documents, combined_name, gender)
            analysis_results = self.analyze_topics(model, combined_name, gender)
            self.generate_visualizations(model, analysis_results, combined_name, gender)
            self.save_analysis_results(analysis_results, combined_name, gender)
            self.save_model(model, combined_name, gender)
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

