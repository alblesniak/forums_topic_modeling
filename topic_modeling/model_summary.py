#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI: Wczytuje zapisany model Top2Vec i generuje skrócony JSON z informacjami o modelu.

Wynik zawiera:
- podstawowe metadane (forum_name/gender jeśli dostępne w modelu/ścieżce, liczba tematów, łączna liczba dokumentów),
- listę tematów (do limitu, domyślnie 25), dla każdego:
  - 20 słów kluczowych (kolejność jak z Top2Vec),
  - 5 najbardziej reprezentatywnych dokumentów (id + treść skrócona),
bez wyników word_scores.

Uwaga: Skrypt zakłada, że model zapisano metodą Top2Vec.save(str(path)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from top2vec import Top2Vec
import sqlite3


def _infer_forum_and_gender_from_path(model_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Próbuje odczytać forum_name oraz gender ze struktury katalogów.

    Oczekiwana konwencja (jak w projekcie):
    data/topics/models/YYYYMMDD/<GENDER>/<FORUMS_CODE>/<RUN_ID>/model
    Zwraca (forum_name, gender) lub (None, None) w razie niepowodzenia.
    """
    try:
        parts = list(model_path.resolve().parts)
    except Exception:
        parts = list(model_path.parts)

    # Szukamy segmentu "models" i dalej 3 następnych: date, gender, forums_code
    try:
        idx = parts.index("models")
        date_str = parts[idx + 1] if len(parts) > idx + 1 else None
        gender = parts[idx + 2] if len(parts) > idx + 2 else None
        forums_code = parts[idx + 3] if len(parts) > idx + 3 else None
        # forum_name nie jest wprost – używamy forums_code jako przybliżenia
        forum_name = forums_code
        return forum_name, gender
    except Exception:
        return None, None


def _infer_run_dir_from_model_path(model_path: Path) -> Optional[Path]:
    """Zwraca przewidywany katalog run odpowiadający modelowi.

    Zamienia segment 'models' na 'runs' i usuwa końcowy element 'model'.
    """
    try:
        parts = list(model_path.resolve().parts)
    except Exception:
        parts = list(model_path.parts)
    try:
        idx = parts.index("models")
        # ścieżka: .../models/YYYYMMDD/GENDER/FORUMS_CODE/TIME/model
        # chcemy: .../runs/YYYYMMDD/GENDER/FORUMS_CODE/TIME
        head = parts[:idx] + ["runs"] + parts[idx + 1:]
        # usuń końcowe "model" jeśli jest
        if head and head[-1] == "model":
            head = head[:-1]
        return Path(*head)
    except Exception:
        return None


def _call_with_reduced(model: Top2Vec, func_name: str, prefer_reduced: bool = False, *args, **kwargs):
    """Próbuje wywołać metodę z reduced=True jeśli dostępne i proszone.
    W przeciwnym razie wywołuje bez tego parametru.
    """
    func = getattr(model, func_name)
    if prefer_reduced:
        try:
            return func(*args, reduced=True, **kwargs)
        except TypeError:
            return func(*args, **kwargs)
    return func(*args, **kwargs)


def summarize_model(
    model_path: Path,
    output_path: Optional[Path] = None,
    max_topics: int = 25,
    words_per_topic: int = 20,
    docs_per_topic: int = 5,
    prefer_reduced: bool = False,
    prefer_original_text: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Wczytuje model Top2Vec i buduje słownik z wymaganym podsumowaniem."""
    model = Top2Vec.load(str(model_path))

    try:
        num_topics = _call_with_reduced(model, "get_num_topics", prefer_reduced=prefer_reduced)
    except Exception:
        num_topics = model.get_num_topics()

    try:
        sizes, size_topic_nums = _call_with_reduced(model, "get_topic_sizes", prefer_reduced=prefer_reduced)
        topic_sizes_map = {int(tn): int(sz) for sz, tn in zip(sizes, size_topic_nums)}
    except Exception:
        try:
            sizes, size_topic_nums = model.get_topic_sizes()
            topic_sizes_map = {int(tn): int(sz) for sz, tn in zip(sizes, size_topic_nums)}
        except Exception:
            topic_sizes_map = {}

    total_documents: Optional[int] = None
    try:
        total_documents = int(np.sum(sizes)) if isinstance(sizes, (list, np.ndarray)) else None
    except Exception:
        total_documents = None

    # Pobierz słowa tematów
    try:
        got = _call_with_reduced(model, "get_topics", prefer_reduced=prefer_reduced)
        if isinstance(got, tuple) and len(got) == 4:
            topic_words_all, _word_scores_all, _topic_scores_all, topic_nums = got
        elif isinstance(got, tuple) and len(got) == 3:
            topic_words_all, _word_scores_all, topic_nums = got
        else:
            topic_words_all, _word_scores_all, topic_nums = model.get_topics()
    except Exception:
        # Fallback: tematy po kolei
        topic_nums = list(range(num_topics))
        topic_words_all = []
        for tnum in topic_nums:
            try:
                words, _scores = model.get_topic_words(tnum)
                topic_words_all.append(words)
            except Exception:
                topic_words_all.append([])

    # Wnioskowanie metadanych z path
    forum_name, gender = _infer_forum_and_gender_from_path(model_path)

    # Przygotowanie źródła oryginalnych tekstów (baza) jeśli dostępne i pożądane
    conn: Optional[sqlite3.Connection] = None
    if prefer_original_text:
        try:
            inferred_run_dir = _infer_run_dir_from_model_path(model_path)
            run_meta_path = inferred_run_dir.joinpath("run.json") if inferred_run_dir else None
            candidate_db_path: Optional[Path] = None
            if db_path is not None:
                candidate_db_path = Path(db_path)
            elif run_meta_path and run_meta_path.exists():
                try:
                    with open(run_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    dbp = meta.get("database_path")
                    if dbp:
                        candidate_db_path = Path(dbp)
                except Exception:
                    candidate_db_path = None
            if candidate_db_path and candidate_db_path.exists():
                conn = sqlite3.connect(str(candidate_db_path))
        except Exception:
            conn = None

    result: Dict[str, Any] = {
        "forum_name": forum_name,
        "gender": gender,
        "num_topics": int(num_topics),
        "total_documents": total_documents,
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "topics": [],
    }

    # Kolejność tematów: jak zwraca Top2Vec.get_topics (topic_nums)
    # Ograniczamy do max_topics
    limited = list(zip(topic_nums, topic_words_all))[:max_topics]

    for idx, (tnum, words) in enumerate(limited):
        # Słowa
        if hasattr(words, "tolist"):
            words_list = words.tolist()
        else:
            words_list = list(words)
        words_list = words_list[:words_per_topic]

        # Dokumenty reprezentatywne
        top_docs: List[Dict[str, Any]] = []
        try:
            # liczba kandydatów do uśrednienia – większa niż chcemy zwrócić
            num_docs_in_topic = topic_sizes_map.get(int(tnum), 0)
            num_candidates = max(docs_per_topic, min(200, max(20, num_docs_in_topic)))
            docs, doc_scores, doc_ids = model.search_documents_by_topic(int(tnum), num_docs=num_candidates)
            # Posortuj malejąco po doc_scores (jeśli dostępne)
            if doc_scores is not None:
                try:
                    scores_list = doc_scores.tolist() if hasattr(doc_scores, "tolist") else list(doc_scores)
                except Exception:
                    scores_list = list(doc_scores)
                order = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
            else:
                order = list(range(len(docs)))
            # Budujemy top N po sortowaniu
            top_indices = order[:docs_per_topic]
            selected_ids: List[Optional[str]] = [doc_ids[i] if doc_ids is not None else None for i in top_indices]
            # Jeśli mamy dostęp do bazy i preferujemy oryginalny tekst – pobierz hurtowo
            original_texts: Dict[str, str] = {}
            if conn is not None and any(selected_ids):
                try:
                    valid_ids: List[int] = []
                    id_map: Dict[int, str] = {}
                    for sid in selected_ids:
                        if sid is None:
                            continue
                        try:
                            iid = int(sid)
                            valid_ids.append(iid)
                            id_map[iid] = sid
                        except Exception:
                            continue
                    if valid_ids:
                        placeholders = ",".join(["?"] * len(valid_ids))
                        cur = conn.cursor()
                        cur.execute(f"SELECT id, content FROM forum_posts WHERE id IN ({placeholders})", valid_ids)
                        for row in cur.fetchall():
                            oid, ocontent = row
                            sid = id_map.get(int(oid))
                            if sid is not None:
                                original_texts[sid] = str(ocontent) if ocontent is not None else ""
                except Exception:
                    original_texts = {}
            # Złóż odpowiedzi – oryginalny tekst jeśli dostępny, w przeciwnym razie przetworzony z modelu
            for i in top_indices:
                d_id = doc_ids[i] if doc_ids is not None else None
                d_content = docs[i]
                text: str
                if d_id is not None and d_id in original_texts:
                    text = original_texts[d_id]
                else:
                    text = str(d_content) if d_content is not None else ""
                text_short = text[:400] + ("…" if len(text) > 400 else "")
                item = {"id": str(d_id) if d_id is not None else None, "text": text_short}
                top_docs.append(item)
        except Exception:
            top_docs = []

        result["topics"].append({
            "topic_id": int(tnum),
            "words": words_list,
            "num_documents": int(topic_sizes_map.get(int(tnum), 0)),
            "top_documents": top_docs,
        })

    # Zapis do pliku jeśli poproszono
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Zamknij połączenie z bazą jeśli otwarte
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generuje skrócony JSON z modelu Top2Vec")
    p.add_argument("--model-path", type=str, default="data/topics/models/20250827/K/ALL/155429_038844/model", help="Ścieżka do modelu Top2Vec (plik lub katalog 'model')")
    p.add_argument("--out", type=str, default=None, help="Ścieżka wynikowego pliku JSON (opcjonalnie)")
    p.add_argument("--max-topics", type=int, default=25, help="Maksymalna liczba tematów do ujęcia")
    p.add_argument("--words-per-topic", type=int, default=20, help="Liczba słów kluczowych na temat")
    p.add_argument("--docs-per-topic", type=int, default=5, help="Liczba dokumentów na temat")
    p.add_argument("--reduced", action="store_true", help="Preferuj przestrzeń zredukowaną przy pobieraniu tematów")
    p.add_argument("--db-path", type=str, default=None, help="Opcjonalna ścieżka do bazy SQLite (nadpisze wykrytą z run.json)")
    p.add_argument("--use-processed-text", action="store_true", help="Wymuś użycie tekstu przetworzonego (bez interpunkcji)")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_path = model_path
    elif model_path.exists():
        # dopuszczamy zarówno katalog, jak i plik (np. bez rozszerzenia)
        pass
    else:
        raise FileNotFoundError(f"Nie znaleziono modelu pod ścieżką: {model_path}")

    out_path = Path(args.out) if args.out else None
    prefer_reduced = bool(args.reduced)
    prefer_original_text = not bool(args.use_processed_text)
    db_override = Path(args.db_path) if args.db_path else None

    result = summarize_model(
        model_path=model_path,
        output_path=out_path,
        max_topics=args.max_topics,
        words_per_topic=args.words_per_topic,
        docs_per_topic=args.docs_per_topic,
        prefer_reduced=prefer_reduced,
        prefer_original_text=prefer_original_text,
        db_path=db_override,
    )

    # Jeśli nie zapisano do pliku, wypisz na stdout
    if out_path is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


