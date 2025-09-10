#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generuje JSON z top-50 słowami i top-50 dokumentami (pełna treść z bazy)
dla wskazanych tematów Top2Vec.

Wejście (argumenty CLI):
--model-path  ścieżka do zapisanego modelu Top2Vec (plik/katalog "model")
--db-path     ścieżka do bazy SQLite (np. merged_forums.db)
--topics      lista id tematów rozdzielona przecinkami (np. 2,11)
--out         ścieżka pliku wyjściowego JSON

Wyjście JSON:
{
  "model_path": str,
  "topics": [
    {
      "topic_id": int,
      "words": [str, ... 50],
      "documents": [
        {
          "post_id": str,
          "score": float,
          "user_id": str|None,
          "username": str|None,
          "forum": str|None,
          "post_date": str|None,
          "url": str|None,
          "content": str|None
        }, ... 50
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sqlite3

from top2vec import Top2Vec


def _get_topic_words(model: Top2Vec, topic_id: int, top_k: int = 50) -> List[str]:
    try:
        topic_words, _word_scores, topic_nums = model.get_topics()
        topic_nums_list = topic_nums.tolist() if hasattr(topic_nums, "tolist") else list(topic_nums)
        topic_words_list = topic_words.tolist() if hasattr(topic_words, "tolist") else list(topic_words)
        topic_nums_list = [int(t) for t in topic_nums_list]
        idx = topic_nums_list.index(int(topic_id))
        words_for_topic = topic_words_list[idx]
        words_seq = words_for_topic.tolist() if hasattr(words_for_topic, "tolist") else list(words_for_topic)
        return [str(w) for w in words_seq[:top_k]]
    except Exception:
        # Fallback
        try:
            words, _scores = model.get_topic_words(int(topic_id), num_words=top_k)
            words_seq = words.tolist() if hasattr(words, "tolist") else list(words)
            return [str(w) for w in words_seq[:top_k]]
        except Exception:
            return []


def _get_top_docs(model: Top2Vec, topic_id: int, top_n: int = 50):
    docs, scores, doc_ids = model.search_documents_by_topic(int(topic_id), num_docs=int(top_n))
    # Konwersje
    ids_list = doc_ids.tolist() if hasattr(doc_ids, "tolist") else list(doc_ids)
    scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
    # Sort malejąco po score (jeśli nieposortowane)
    order = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
    ids_sorted = [str(ids_list[i]) for i in order][:top_n]
    scores_sorted = [float(scores_list[i]) for i in order][:top_n]
    return ids_sorted, scores_sorted


def _fetch_posts(conn: sqlite3.Connection, post_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not post_ids:
        return {}
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
    cur = conn.cursor()
    cur.execute(query, post_ids)
    rows = cur.fetchall()
    # Zbuduj mapę id -> dane
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        post_id, user_id, username, forum, content, post_date, url = row
        result[str(post_id)] = {
            "post_id": str(post_id),
            "user_id": None if user_id is None else str(user_id),
            "username": None if username is None else str(username),
            "forum": None if forum is None else str(forum),
            "content": None if content is None else str(content),
            "post_date": None if post_date is None else str(post_date),
            "url": None if url is None else str(url),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Dumpuje słowa i dokumenty dla wskazanych tematów Top2Vec")
    ap.add_argument("--model-path", required=True, type=str)
    ap.add_argument("--db-path", required=True, type=str)
    ap.add_argument("--topics", required=True, type=str, help="Lista ID tematów, np. 2,11")
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    db_path = Path(args.db_path)
    out_path = Path(args.out)
    topic_ids = [int(t.strip()) for t in args.topics.split(",") if t.strip()]

    model = Top2Vec.load(str(model_path))

    conn = sqlite3.connect(str(db_path))
    try:
        result: Dict[str, Any] = {
            "model_path": str(model_path),
            "topics": [],
        }
        for t_id in topic_ids:
            words = _get_topic_words(model, t_id, top_k=50)
            ids, scores = _get_top_docs(model, t_id, top_n=50)
            posts_map = _fetch_posts(conn, ids)
            docs = []
            for pid, sc in zip(ids, scores):
                meta = posts_map.get(pid, {"post_id": pid})
                docs.append({
                    "post_id": pid,
                    "score": sc,
                    "user_id": meta.get("user_id"),
                    "username": meta.get("username"),
                    "forum": meta.get("forum"),
                    "post_date": meta.get("post_date"),
                    "url": meta.get("url"),
                    "content": meta.get("content"),
                })
            result["topics"].append({
                "topic_id": int(t_id),
                "words": words,
                "documents": docs,
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


