#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator chmur słów i wykresów timeline dla zapisanych modeli Top2Vec.

Wejście: ścieżki do modeli Top2Vec (plik/katalog "model") oraz baza merged_forums.db
Wyjście: obrazy PNG zapisane do presentation/data/topics/results/visualizations/K i M

Uwaga: Skrypt nie wymaga R. Wykorzystuje matplotlib i wordcloud.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw
from top2vec import Top2Vec
from wordcloud import WordCloud

# Paleta kolorów dopasowana do oryginalnych wizualizacji
COLOR_PALETTE = [
    (47, 62, 70),      # ciemny szary/niebieski
    (242, 92, 84),     # czerwony
    (122, 158, 127),   # zielony
    (148, 62, 119),    # fioletowy
    (80, 125, 188),    # niebieski
    (74, 48, 109),     # ciemny fioletowy
    (222, 164, 126),   # brązowy/beżowy
    (193, 163, 163),   # różowy/szary
    (208, 88, 113),    # różowy
    (95, 95, 95),      # szary
    (166, 38, 57),     # ciemny czerwony
    (142, 65, 98),     # ciemny różowy
    (61, 90, 128),     # ciemny niebieski
    (110, 106, 111),   # szary
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_topic_ids(model: Top2Vec) -> List[int]:
    try:
        got = model.get_topics()
        if isinstance(got, tuple):
            # 3- lub 4-elementowe zwrotki zależnie od wersji Top2Vec
            if len(got) == 4:
                _tw, _ws, _ts, topic_nums = got
            else:
                _tw, _ws, topic_nums = got
            topic_nums_list = topic_nums.tolist() if hasattr(topic_nums, 'tolist') else list(topic_nums)
            return [int(t) for t in topic_nums_list]
    except Exception:
        pass
    try:
        return list(range(int(model.get_num_topics())))
    except Exception:
        return []


def _get_topic_words_scores(model: Top2Vec, topic_id: int, top_k: int = 150) -> Tuple[List[str], List[float]]:
    # Preferuj API z parametrem num_words, aby zwiększyć liczbę słów (np. do 75)
    try:
        words, scores = model.get_topic_words(int(topic_id), num_words=int(top_k))
        words_list = (words.tolist() if hasattr(words, 'tolist') else list(words))[:top_k]
        scores_list = (scores.tolist() if hasattr(scores, 'tolist') else list(scores))[:top_k]
        return [str(w) for w in words_list], [float(s) for s in scores_list]
    except Exception:
        pass
    # Fallback do get_topics (często zwraca stałą liczbę, np. 50)
    try:
        got = model.get_topics()
        if isinstance(got, tuple):
            if len(got) == 4:
                topic_words_all, word_scores_all, _topic_scores_all, topic_nums = got
            else:
                topic_words_all, word_scores_all, topic_nums = got
            topic_nums_list = topic_nums.tolist() if hasattr(topic_nums, 'tolist') else list(topic_nums)
            idx = topic_nums_list.index(int(topic_id))
            words = topic_words_all[idx]
            scores = word_scores_all[idx]
            words_list = (words.tolist() if hasattr(words, 'tolist') else list(words))[:top_k]
            scores_list = (scores.tolist() if hasattr(scores, 'tolist') else list(scores))[:top_k]
            return [str(w) for w in words_list], [float(s) for s in scores_list]
    except Exception:
        pass
    return [], []


def _get_topic_size(model: Top2Vec, topic_id: int) -> int:
    try:
        sizes, topic_nums = model.get_topic_sizes()
        tnums = topic_nums.tolist() if hasattr(topic_nums, 'tolist') else list(topic_nums)
        sizes_list = sizes.tolist() if hasattr(sizes, 'tolist') else list(sizes)
        mapping = {int(t): int(s) for s, t in zip(sizes_list, tnums)}
        return int(mapping.get(int(topic_id), 0))
    except Exception:
        try:
            return int(model.get_num_topics())  # bezpieczny fallback; nie jest to realny rozmiar tematu
        except Exception:
            return 0


def _get_topic_documents(model: Top2Vec, topic_id: int, max_docs: Optional[int] = None) -> Tuple[List[str], List[float]]:
    # Ustal liczbę dokumentów: weź rozmiar tematu, ale ogranicz rozsądnie (np. do 50k)
    size = _get_topic_size(model, topic_id)
    if size <= 0:
        size = 1000
    num_docs = min(size, max_docs) if isinstance(max_docs, int) and max_docs > 0 else size
    num_docs = int(max(1, min(num_docs, 50000)))
    docs, scores, doc_ids = model.search_documents_by_topic(int(topic_id), num_docs=num_docs)
    ids_list = doc_ids.tolist() if hasattr(doc_ids, 'tolist') else list(doc_ids)
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    return [str(x) for x in ids_list], [float(x) for x in scores_list]


def _fetch_years_for_post_ids(conn: sqlite3.Connection, post_ids: List[str]) -> Dict[str, int]:
    if not post_ids:
        return {}
    id_to_year: Dict[str, int] = {}
    # Batchuj, żeby uniknąć zbyt dużych zapytań IN
    batch_size = 1000
    for i in range(0, len(post_ids), batch_size):
        batch = post_ids[i:i + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        query = f"""
            SELECT id, post_date
            FROM forum_posts
            WHERE id IN ({placeholders})
        """
        cur = conn.cursor()
        cur.execute(query, batch)
        rows = cur.fetchall()
        for pid, pdate in rows:
            year = None
            try:
                # post_date bywa 'YYYY-mm-dd HH:MM:SS' lub 'YYYY-mm-dd' lub samo 'YYYY'
                s = str(pdate) if pdate is not None else ""
                if len(s) >= 4 and s[:4].isdigit():
                    year = int(s[:4])
            except Exception:
                year = None
            if year is not None:
                id_to_year[str(pid)] = year
    return id_to_year


def _build_year_series(years: List[int], weights: Optional[List[float]] = None) -> pd.Series:
    if not years:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"year": years})
    if weights is None:
        ser = df["year"].value_counts().sort_index()
        ser.name = "count"
        return ser
    else:
        df["weight"] = weights
        ser = df.groupby("year")["weight"].sum().sort_index()
        ser.name = "weight"
        return ser


def _plot_timeline(year_series: pd.Series, out_path: Path, title: str) -> None:
    if year_series.empty:
        # wygeneruj pusty wykres z komunikatem
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "Brak danych", ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        return

    x = list(year_series.index)
    y = list(year_series.values)

    plt.figure(figsize=(12, 7))
    plt.bar(x, y, color="#507DBC", alpha=0.5)
    # Prosta krzywa wygładzająca przez średnią kroczącą (okno 3) – bez zewnętrznych zależności
    try:
        y_arr = np.array(y, dtype=float)
        if len(y_arr) >= 3:
            kernel = np.ones(3) / 3.0
            smooth = np.convolve(y_arr, kernel, mode='same')
            plt.plot(x, smooth, color="#274060", linewidth=2)
    except Exception:
        pass

    plt.xlabel("Rok")
    plt.ylabel("Liczba/masa dokumentów")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _detect_bold_font_path(preferred_path: Optional[str] = None) -> Optional[str]:
    # 1) Preferuj ścieżkę z CLI, jeśli istnieje
    if preferred_path:
        p = Path(preferred_path)
        if p.exists():
            return str(p)
    # 2) Spróbuj odnaleźć czcionkę o wadze 'bold' poprzez matplotlib
    try:
        fp = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
        if fp and Path(fp).exists():
            return fp
    except Exception:
        pass
    # 3) Typowe ścieżki dla macOS
    common = [
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for c in common:
        if Path(c).exists():
            return c
    return None


def _get_presentation_bg_color(css_path: Path = Path("presentation/css/theme/custom.css")) -> str:
    default = "#FAFAF8"
    try:
        if css_path.exists():
            text = css_path.read_text(encoding="utf-8", errors="ignore")
            # proste wyszukiwanie zmiennej --bg-ivory: #XXXXXX;
            import re
            m = re.search(r"--bg-ivory:\s*(#[0-9A-Fa-f]{6})", text)
            if m:
                return m.group(1)
    except Exception:
        pass
    return default


def _make_rounded_rect_mask(width: int, height: int, radius: int) -> np.ndarray:
    # Tworzy maskę: białe (255) w środku zaokrąglonego prostokąta, czarne (0) na rogach
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    # PIL rounded_rectangle dostępne w nowszych wersjach Pillow
    try:
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    except Exception:
        # Fallback: ręczne rysowanie (prosty prostokąt bez zaokrągleń)
        draw.rectangle((0, 0, width, height), fill=255)
    return np.array(img)


def _plot_wordcloud(word_weights: Dict[str, float], out_path: Path, topic_id: int, font_path: Optional[str], background_color: str, rounded_corners: bool, corner_radius: int) -> None:
    # Wybierz kolor główny per temat (cyklicznie)
    main_color = COLOR_PALETTE[int(topic_id) % len(COLOR_PALETTE)]

    def single_color_func(word, font_size, position, orientation, font_path, random_state):
        return main_color

    width = 1600
    height = 1200
    mask = _make_rounded_rect_mask(width, height, corner_radius) if rounded_corners else None

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=150,
        colormap=None,
        color_func=single_color_func,
        font_path=font_path,
        font_step=1,
        max_font_size=200,
        min_font_size=20,
        relative_scaling=0.5,
        prefer_horizontal=0.9,
        margin=10,
        scale=1,
        collocations=False,
        include_numbers=False,
        min_word_length=1,
        stopwords=set(),
        random_state=42,
        collocation_threshold=30,
        normalize_plurals=True,
        mode='RGB',
        mask=mask,
    )

    wc.generate_from_frequencies(word_weights)
    # Fallback, gdyby nic nie narysowało (np. po filtrach)
    try:
        layout = getattr(wc, 'layout_', [])
        if not getattr(wc, 'words_', None) or not layout:
            wc = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                max_words=max(50, min(200, len(word_weights) or 50)),
                colormap=None,
                color_func=single_color_func,
                font_path=font_path,
                font_step=1,
                max_font_size=220,
                min_font_size=10,
                relative_scaling=0.5,
                prefer_horizontal=1.0,
                margin=2,
                scale=1,
                collocations=False,
                include_numbers=True,
                min_word_length=1,
                stopwords=set(),
                random_state=42,
                normalize_plurals=True,
                mode='RGB',
                mask=None,
            )
            wc.generate_from_frequencies(word_weights)
    except Exception:
        pass
    _ensure_dir(out_path.parent)
    wc.to_file(str(out_path))


def process_model(model_path: Path, gender_letter: str, output_root: Path, database_path: Path, use_weights: bool = False, bold_font_path: Optional[str] = None, background_color: Optional[str] = None, rounded_corners: bool = True, corner_radius: int = 40) -> None:
    model: Top2Vec = Top2Vec.load(str(model_path))

    # Wyjściowe katalogi
    base_dir = output_root / gender_letter
    wc_dir = base_dir / "wordclouds"
    tl_dir = base_dir / "timelines"
    _ensure_dir(wc_dir)
    _ensure_dir(tl_dir)

    # Czcionka pogrubiona dla WordCloud
    detected_bold_font = _detect_bold_font_path(bold_font_path)
    bg_color = background_color or _get_presentation_bg_color()

    topic_ids = _get_topic_ids(model)
    if not topic_ids:
        print(f"Brak tematów w modelu: {model_path}")
        return

    # Połączenie z bazą
    conn = sqlite3.connect(str(database_path))
    try:
        for topic_id in topic_ids:
            # 1) Chmura słów – weź do 75 słów (preferuj API num_words)
            words, scores = _get_topic_words_scores(model, topic_id, top_k=75)
            word_weights = {w: float(s) for w, s in zip(words, scores)}
            wc_out = wc_dir / f"topic_{int(topic_id)}_wordcloud.png"
            try:
                # Log pomocniczy
                print(f"[wordcloud] {gender_letter} topic {int(topic_id)}: input_words={len(word_weights)} sample={(list(word_weights)[:5])}")
            except Exception:
                pass
            _plot_wordcloud(word_weights, wc_out, int(topic_id), detected_bold_font, bg_color, rounded_corners, corner_radius)

            # 2) Timeline (liczność lub ważona) po latach
            doc_ids, doc_scores = _get_topic_documents(model, topic_id, max_docs=None)
            id_to_year = _fetch_years_for_post_ids(conn, doc_ids)
            years: List[int] = []
            weights: Optional[List[float]] = [] if use_weights else None
            for pid, score in zip(doc_ids, doc_scores):
                y = id_to_year.get(str(pid))
                if y is None:
                    continue
                years.append(int(y))
                if weights is not None:
                    weights.append(float(score))

            series = _build_year_series(years, weights)
            tl_out = tl_dir / f"topic_{int(topic_id)}_timeline.png"
            title = f"Timeline tematu {int(topic_id)} ({gender_letter})"
            _plot_timeline(series, tl_out, title)

    finally:
        try:
            conn.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generuje chmury słów i timeline'y dla zapisanych modeli Top2Vec")
    ap.add_argument("--model-k", required=False, default="data/topics/models/20250827/K/ALL/155429_038844/model", help="Ścieżka do modelu dla K")
    ap.add_argument("--model-m", required=False, default="data/topics/models/20250827/M/ALL/194903_560595/model", help="Ścieżka do modelu dla M")
    ap.add_argument("--db", required=False, default="data/databases/merged_forums.db", help="Ścieżka do merged_forums.db")
    ap.add_argument("--out", required=False, default="presentation/data/topics/results/visualizations", help="Katalog wyjściowy wizualizacji")
    ap.add_argument("--weights", action="store_true", help="Użyj wag dokumentów do timeline zamiast zliczeń")
    ap.add_argument("--font", required=False, default=None, help="Ścieżka do czcionki pogrubionej (TTF/OTF)")
    ap.add_argument("--bg-color", required=False, default=None, help="Kolor tła (np. #FAFAF8); domyślnie z prezentacji")
    ap.add_argument("--no-rounded", action="store_true", help="Wyłącz zaokrąglone rogi (maskę)")
    ap.add_argument("--corner-radius", type=int, default=40, help="Promień rogów maski (px)")
    ap.add_argument("--skip-k", action="store_true", help="Pomiń generowanie dla K")
    ap.add_argument("--skip-m", action="store_true", help="Pomiń generowanie dla M")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    _ensure_dir(out_root)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"✗ Brak bazy danych: {db_path}")
        return 2

    # K
    if not args.skip_k:
        model_k = Path(args.model_k)
        if model_k.exists():
            print(f"→ Generuję K z modelu: {model_k}")
            process_model(
                model_k,
                "K",
                out_root,
                db_path,
                use_weights=bool(args.weights),
                bold_font_path=args.font,
                background_color=args.bg_color,
                rounded_corners=not args.no_rounded,
                corner_radius=int(args.corner_radius or 40),
            )
        else:
            print(f"⚠️  Model K nie istnieje: {model_k}")

    # M
    if not args.skip_m:
        model_m = Path(args.model_m)
        if model_m.exists():
            print(f"→ Generuję M z modelu: {model_m}")
            process_model(
                model_m,
                "M",
                out_root,
                db_path,
                use_weights=bool(args.weights),
                bold_font_path=args.font,
                background_color=args.bg_color,
                rounded_corners=not args.no_rounded,
                corner_radius=int(args.corner_radius or 40),
            )
        else:
            print(f"⚠️  Model M nie istnieje: {model_m}")

    print(f"✓ Zakończono. Wyniki: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


