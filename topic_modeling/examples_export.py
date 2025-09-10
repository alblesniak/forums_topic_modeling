#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksport przykładów dokumentów z wybranego tematu modelu Top2Vec do Excela.

Wejście:
- ścieżka do modelu (katalog zawierający zapisany model Top2Vec)
- numer tematu (int)
- liczba dokumentów (Top N wg score w temacie)

Wyjście:
- plik XLSX w katalogu wyników danego modelu, w podfolderze examples/
  nazwa: topic_<N>_<k1>_<k2>_<k3>.xlsx gdzie k1..k3 to top-3 słowa kluczowe tematu

W Excelu zapisujemy kolumny:
- post_id, user_id, username, forum, content, word_score, url

Struktura katalogów wyników jest zgodna z tą tworzoną przez TopicModelingAnalyzer
(wyniki w analysis/topic_modeling/topic_modeling_script.py), tzn. w drzewie
data/topics/results/<YYYYMMDD>/<GENDER>/<FORUMS_CODE>/<HHMMSS>/examples/
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from top2vec import Top2Vec


@dataclass
class ExportParams:
    model_path: Path
    database_path: Path
    results_dir: Path
    topic_num: int
    top_n: Optional[int]


def _sanitize_token(token: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ_-]+", "", token).strip("-_ ")
    return token[:24] if token else "kw"


def _infer_results_dir_from_model(model_path: Path) -> Path | None:
    # Model zapisywany jest jako <models_dir>/model (plik lub folder)
    # Chcemy pobrać równoległy katalog results na tym samym poziomie drzewa
    # .../data/topics/models/<date>/<gender>/<forums>/<time>/model
    # -> .../data/topics/results/<date>/<gender>/<forums>/<time>
    parts = list(model_path.resolve().parts)
    try:
        idx = parts.index("models")
    except ValueError:
        return None
    # Zastąp "models" -> "results" i usuń końcowy element "model" jeśli występuje
    if parts[-1] == "model":
        parts = parts[:-1]
    parts[idx] = "results"
    return Path(*parts)


def _load_model(model_path: Path) -> Top2Vec:
    return Top2Vec.load(str(model_path))


def _get_topic_keywords(model: Top2Vec, topic_num: int, top_k: int = 3) -> List[str]:
    # Zgodność z API Top2Vec: pobierz wszystkie tematy i wybierz po topic_num
    try:
        topic_words, word_scores, topic_nums = model.get_topics()
        # Konwersje typów (numpy -> list)
        topic_nums_list = topic_nums.tolist() if hasattr(topic_nums, "tolist") else list(topic_nums)
        topic_words_list = topic_words.tolist() if hasattr(topic_words, "tolist") else list(topic_words)
        # Upewnij się, że to listy prostych typów
        topic_nums_list = [int(t) for t in topic_nums_list]
        # Znajdź indeks danego tematu
        idx = topic_nums_list.index(int(topic_num))
        words_for_topic = topic_words_list[idx]
        words_seq = words_for_topic.tolist() if hasattr(words_for_topic, "tolist") else list(words_for_topic)
        return [str(w) for w in words_seq[:top_k]]
    except Exception:
        return []


def _get_top_documents(model: Top2Vec, topic_num: int, top_n: Optional[int]) -> Tuple[List[str], List[float]]:
    # Zwraca listę document_ids i odpowiadających im score
    # Jeśli top_n jest None, pobierz rozmiar tematu i użyj go jako limitu
    num_docs: int
    if top_n is None:
        try:
            topic_sizes, topic_nums = model.get_topic_sizes()
            # mapuj numer tematu -> rozmiar
            sizes_map = {int(tn): int(sz) for sz, tn in zip(topic_sizes, topic_nums)}
            num_docs = max(1, sizes_map.get(int(topic_num), 1))
        except Exception:
            # Fallback gdy API niedostępne: weź duży limit
            num_docs = 1_000_000
    else:
        num_docs = int(top_n)

    docs, scores, doc_ids = model.search_documents_by_topic(int(topic_num), num_docs=num_docs)
    # Niektóre implementacje mogą zwrócić None/ndarray; prosty fallback
    ids = [str(d) for d in (doc_ids.tolist() if hasattr(doc_ids, "tolist") else list(doc_ids))]
    sc = [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else list(scores))]
    return ids, sc


def _fetch_post_rows(conn: sqlite3.Connection, post_ids: List[str]) -> pd.DataFrame:
    if not post_ids:
        return pd.DataFrame(columns=["post_id", "user_id", "username", "forum", "content", "post_date", "url"])
    placeholders = ",".join(["?"] * len(post_ids))
    query = f"""
        SELECT
            fp.id AS post_id,
            fp.user_id AS user_id,
            fp.username AS username,
            f.spider_name AS forum,
            fp.content AS content,
            fp.post_date AS post_date,
            fp.url AS url
        FROM forum_posts fp
        JOIN forum_threads ft ON fp.thread_id = ft.id
        JOIN forum_sections fs ON ft.section_id = fs.id
        JOIN forums f ON fs.forum_id = f.id
        WHERE fp.id IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=post_ids)
    return df


_INVALID_XLS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def _sanitize_excel_text(value: Optional[object]) -> Optional[str]:
    """Usuwa nielegalne znaki XML/Excel z tekstu.

    openpyxl odrzuca kontrolne znaki 0x00-0x1F (z wyj. \t, \n, \r).
    """
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        return None
    return _INVALID_XLS_RE.sub("", text)


def export_examples(params: ExportParams) -> Path:
    model = _load_model(params.model_path)

    # Słowa kluczowe do nazwy pliku
    top_words = _get_topic_keywords(model, params.topic_num, top_k=3)
    safe_words = [_sanitize_token(w) for w in top_words]

    # Pobierz dokumenty
    doc_ids, doc_scores = _get_top_documents(model, params.topic_num, params.top_n)

    # Pobierz z bazy metadane postów
    conn = sqlite3.connect(str(params.database_path))
    try:
        df_posts = _fetch_post_rows(conn, doc_ids)
    finally:
        conn.close()

    # Dołącz score wg pozycji id -> score
    score_map = {pid: sc for pid, sc in zip(doc_ids, doc_scores)}
    df_posts["word_score"] = df_posts["post_id"].astype(str).map(score_map).fillna(0.0)

    # Uporządkuj kolejność i sortuj malejąco po score
    columns = ["post_id", "user_id", "username", "forum", "content", "post_date", "word_score", "url"]
    for col in columns:
        if col not in df_posts.columns:
            df_posts[col] = None
    df_final = df_posts[columns].sort_values(by="word_score", ascending=False).reset_index(drop=True)

    # Katalog docelowy examples/
    results_dir = params.results_dir or _infer_results_dir_from_model(params.model_path)
    if results_dir is None:
        # Fallback: zapis obok modelu
        results_dir = params.model_path.parent.parent / "results"
    examples_dir = Path(results_dir) / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Nazwa pliku
    words_part = "_".join(w for w in safe_words if w)
    filename = f"topic_{params.topic_num}_{words_part}.xlsx" if words_part else f"topic_{params.topic_num}.xlsx"
    out_path = examples_dir / filename

    # Zapis do Excela
    # Sanitizacja pól tekstowych przed zapisem do Excela
    df_to_write = df_final.copy()
    for col in ["content", "username", "forum", "url"]:
        if col in df_to_write.columns:
            df_to_write[col] = df_to_write[col].map(_sanitize_excel_text)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_to_write.to_excel(writer, index=False, sheet_name="examples")

    return out_path


def cli_export(model_path: str, database_path: str, topic_num: int, top_n: Optional[int], results_dir: str | None = None) -> Path:
    params = ExportParams(
        model_path=Path(model_path),
        database_path=Path(database_path),
        results_dir=Path(results_dir) if results_dir else None,
        topic_num=int(topic_num),
        top_n=(int(top_n) if top_n is not None else None),
    )
    return export_examples(params)


__all__ = ["ExportParams", "export_examples", "cli_export"]


