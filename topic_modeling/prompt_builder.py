#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline generowania taksonomii 3-poziomowej oraz promptu systemowego na podstawie
przykładów z Excela (bez użycia modelu Top2Vec).

Kroki (wysokopoziomowo):
1) Streszczenia i cytaty kluczowe dla postów z Excela
2) Propozycje kategorii głównych per post (LLM, batchowo)
3) Konsolidacja kategorii głównych (deduplikacja, synonimy, limit 8–15)
4) Indukcja podkategorii dla każdej głównej (3–8)
5) Indukcja pod‑podkategorii dla każdej podkategorii (2–8)
6) Składanie i numeracja (1, 1.1, 1.1.1)
7) Przypisania postów do pełnych ścieżek (z confidence)
8) Walidacja pokrycia i opcjonalne korekty
9) Eksport artefaktów (taxonomy.json, prompt.md, assignments.jsonl, aktualizacja Excela)
10) Kompilacja finalnego system_message

UWAGA: Ten moduł korzysta z LLM przez bibliotekę openai.
Wymaga prawidłowego ustawienia klucza i modelu w analysis/config.py (LLM_CONFIG).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    # OpenAI v1.x – użyjemy z base_url OpenRouter
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Konfiguracja LLM z pliku analysis/config.py
try:
    from analysis.config import LLM_CONFIG  # type: ignore
except Exception:  # pragma: no cover
    LLM_CONFIG = {
        'model': 'gpt-4o-mini',
        'temperature': 0.7,
        'max_tokens': 1000,
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
    }


# ====== Pomocnicze struktury danych ======

@dataclass
class BuilderParams:
    excel_path: Path
    output_base_dir: Path
    theme_slug: str
    batch_size: int = 50
    max_posts: Optional[int] = None  # opcjonalne cięcie


@dataclass
class LLMParams:
    model: str
    temperature: float
    max_tokens: int
    api_key: str
    base_url: Optional[str] | None = None
    default_headers: Optional[Dict[str, str]] | None = None


def _make_client(llm: LLMParams):
    if OpenAI is None:
        raise RuntimeError("Biblioteka openai nie jest zainstalowana. Dodaj 'openai' do zależności.")
    if not llm.api_key:
        raise RuntimeError("Brak klucza API dla OpenRouter. Ustaw OPENROUTER_API_KEY lub LLM_CONFIG['api_key'].")
    kwargs = {"api_key": llm.api_key}
    if getattr(llm, "base_url", None):
        kwargs["base_url"] = llm.base_url  # OpenRouter: https://openrouter.ai/api/v1
    if getattr(llm, "default_headers", None):
        kwargs["default_headers"] = llm.default_headers
    return OpenAI(**kwargs)


def _chunks(lst: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [lst]
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _now_slug() -> Tuple[str, str]:
    dt = datetime.now()
    return dt.strftime('%Y%m%d'), dt.strftime('%H%M%S')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_text(x: Any) -> str:
    return (str(x) if x is not None else '').strip()


def _render_prompt_summarize(batch_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Przygotuj content z 1–2 zdaniami i 1 cytatem zwracany jako JSON
    sys = """
Streść każdy post po polsku w 1–2 zdaniach i wybierz 1 kluczowy cytat.
Zwróć WYŁĄCZNIE JSON (bez dodatkowego tekstu) jako listę: [{"post_id":"...","summary":"...","quote":"..."}].
"""
    examples = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps([
            {"post_id": str(r.get('post_id', r.get('id', str(i)))), "content": _safe_text(r.get('content', ''))}
            for i, r in enumerate(batch_rows)
        ], ensure_ascii=False)}
    ]
    return examples


def _render_prompt_suggest_categories(theme: str, batch_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sys = """
Na podstawie streszczeń zaproponuj dla każdego wpisu 1–3 kategorie główne (PL)
jako krótkie nazwy (<= 5 słów). Zwróć JSON listę:
[{"post_id":"...", "candidates":["...", "..."]}].
"""
    inst = (
        f"Temat przewodni: {theme}. Preferuj spójne, ogólne nazwy. Unikaj duplikatów i zbyt szczegółowych etykiet."
    )
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": inst},
        {"role": "user", "content": json.dumps([
            {
                "post_id": str(r.get('post_id', r.get('id', str(i)))),
                "summary": _safe_text(r.get('summary', '')),
                "quote": _safe_text(r.get('quote', '')),
            }
            for i, r in enumerate(batch_rows)
        ], ensure_ascii=False)}
    ]
    return msg


def _render_prompt_consolidate_main(candidates: List[str]) -> List[Dict[str, str]]:
    sys = """
Masz listę kandydatur nazw kategorii głównych (PL). Zdeduplikuj, połącz synonimy,
pozostaw 8–15 zwięzłych kategorii (<= 5 słów), każdą z 1–2 zdaniowym opisem.
Jeżeli to uzasadnione, możesz przeformułować lub dopasować nazwy kategorii,
aby lepiej oddawały spójną grupę treści (bez zmiany ich ogólnego zakresu).
Zwróć WYŁĄCZNIE JSON: [{"name":"...","desc":"..."}].
"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps({"candidates": candidates}, ensure_ascii=False)}
    ]
    return msg


def _render_prompt_induce_children(level_name: str, parent_name: str, examples_rows: List[Dict[str, Any]],
                                   min_k: int, max_k: int) -> List[Dict[str, str]]:
    sys = f"""
Dla kategorii '{parent_name}' zaproponuj {min_k}–{max_k} {level_name} (PL).
Każda: zwięzła nazwa (<= 5 słów) i 1–2 zdaniowy opis.
Zwróć JSON: [{{"name":"...","desc":"..."}}].
"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps([
            {"summary": _safe_text(r.get('summary', '')), "quote": _safe_text(r.get('quote', ''))}
            for r in examples_rows
        ], ensure_ascii=False)}
    ]
    return msg


def _render_prompt_assemble_numbering(taxonomy_tree: Dict[str, Any]) -> List[Dict[str, str]]:
    sys = """
Nadaj numerację 1, 1.1, 1.1.1 dla przekazanej hierarchii (PL). Zwróć JSON:
[{"id":"1","name":"...","children":[{"id":"1.1","name":"...","children":[{"id":"1.1.1","name":"...","desc":"..."}]}]}].
"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(taxonomy_tree, ensure_ascii=False)}
    ]
    return msg


def _render_prompt_assign(taxonomy_json: Any, batch_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sys = """
Masz taksonomię 3‑poziomową (PL). Dla każdego wpisu przypisz pełną ścieżkę 'X.Y.Z'.
Gdy niepewne, użyj 'Inne' na trzecim poziomie w ramach najbliższej sensownej głównej/podkategorii.
Zwróć JSON listę: [{"post_id":"...","path":"1.2.3","confidence":0.0}].
"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps({"taxonomy": taxonomy_json}, ensure_ascii=False)},
        {"role": "user", "content": json.dumps([
            {"post_id": str(r.get('post_id')), "summary": _safe_text(r.get('summary', '')), "quote": _safe_text(r.get('quote', ''))}
            for r in batch_rows
        ], ensure_ascii=False)}
    ]
    return msg


def _render_prompt_compile_system_message(taxonomy_json: Any) -> List[Dict[str, str]]:
    sys = """
Na podstawie taksonomii (PL) wygeneruj kompletny system_message z zasadami:
3 poziomy, obowiązkowa pełna ścieżka, możliwość wielu ścieżek, 'Inne' jako ostateczność,
format wyjściowy nested JSON z numeracją jako kluczami (bez etykiet typu 'Kategoria').
Wypisz pełną listę kategorii z krótkimi wskazówkami. Zwróć czysty tekst promptu.
"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(taxonomy_json, ensure_ascii=False)}
    ]
    return msg


def _local_numbering(tree: Any) -> List[Dict[str, Any]]:
    """Nadaje identyfikatory 1, 1.1, 1.1.1 lokalnie (bez LLM).
    Oczekuje struktury {"children": [...]} lub listy węzłów na poziomie głównym.
    Każdy węzeł: {"name": str, "desc": str?, "children": list?}
    """
    if isinstance(tree, dict) and isinstance(tree.get("children"), list):
        roots = list(tree.get("children", []))
    elif isinstance(tree, list):
        roots = list(tree)
    else:
        roots = []

    def safe_children(node: Any) -> List[Dict[str, Any]]:
        ch = node.get("children") if isinstance(node, dict) else None
        return list(ch) if isinstance(ch, list) else []

    numbered: List[Dict[str, Any]] = []
    for i, root in enumerate(roots, start=1):
        name = _safe_text(root.get("name", "")) if isinstance(root, dict) else ""
        desc = _safe_text(root.get("desc", "")) if isinstance(root, dict) else ""
        subnodes = safe_children(root)
        new_root: Dict[str, Any] = {"id": f"{i}", "name": name}
        if desc:
            new_root["desc"] = desc
        new_root_children: List[Dict[str, Any]] = []
        for j, sub in enumerate(subnodes, start=1):
            sub_name = _safe_text(sub.get("name", "")) if isinstance(sub, dict) else ""
            sub_desc = _safe_text(sub.get("desc", "")) if isinstance(sub, dict) else ""
            subsubnodes = safe_children(sub)
            new_sub: Dict[str, Any] = {"id": f"{i}.{j}", "name": sub_name}
            if sub_desc:
                new_sub["desc"] = sub_desc
            new_sub_children: List[Dict[str, Any]] = []
            for k, subsub in enumerate(subsubnodes, start=1):
                ss_name = _safe_text(subsub.get("name", "")) if isinstance(subsub, dict) else ""
                ss_desc = _safe_text(subsub.get("desc", "")) if isinstance(subsub, dict) else ""
                new_sub_children.append({
                    "id": f"{i}.{j}.{k}",
                    "name": ss_name,
                    "desc": ss_desc,
                })
            if new_sub_children:
                new_sub["children"] = new_sub_children
            new_root_children.append(new_sub)
        if new_root_children:
            new_root["children"] = new_root_children
        numbered.append(new_root)
    return numbered


def _chat(client, llm: LLMParams, messages: List[Dict[str, str]]) -> str:
    # Ujednolicony interfejs wywołań do modelu chat
    resp = client.chat.completions.create(
        model=llm.model,
        temperature=llm.temperature,
        messages=messages,
        max_tokens=llm.max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_json_safe(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # Spróbuj wydobyć blok JSON jeżeli LLM dodał komentarze
        start = text.find('{')
        arr = text.find('[')
        if start == -1 and arr == -1:
            return None
        s = min([x for x in [start, arr] if x != -1])
        end = text.rfind(']')
        end2 = text.rfind('}')
        e = max(end, end2)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
        return None


class TaxonomyPromptBuilder:
    def __init__(self, params: BuilderParams, llm: Optional[LLMParams] = None):
        self.params = params
        if llm is None:
            # Klucz preferencyjnie z OPENROUTER_API_KEY
            cfg_key = str(LLM_CONFIG.get('api_key', '')).strip() if isinstance(LLM_CONFIG, dict) else ''
            env_key = os.environ.get('OPENROUTER_API_KEY', '').strip() or os.environ.get('OPENAI_API_KEY', '').strip()
            resolved_key = cfg_key or env_key
            base_url = str(LLM_CONFIG.get('base_url', '')).strip() if isinstance(LLM_CONFIG, dict) else ''
            # Opcjonalne nagłówki OpenRouter (dla statystyk/identyfikacji aplikacji)
            default_headers: Dict[str, str] = {}
            ref = os.environ.get('OPENROUTER_HTTP_REFERER', '').strip()
            title = os.environ.get('OPENROUTER_APP_TITLE', '').strip()
            if ref:
                default_headers['HTTP-Referer'] = ref
            if title:
                default_headers['X-Title'] = title
            if not default_headers:
                default_headers = None  # type: ignore
            llm = LLMParams(
                model=str(LLM_CONFIG.get('model', 'gpt-4o-mini')),
                temperature=float(LLM_CONFIG.get('temperature', 0.7)),
                max_tokens=int(LLM_CONFIG.get('max_tokens', 1000)),
                api_key=resolved_key,
                base_url=base_url or None,
                default_headers=default_headers,
            )
        self.llm = llm
        self.client = _make_client(self.llm)

        date_str, time_str = _now_slug()
        base = Path(self.params.output_base_dir)
        self.run_dir = base / "taxonomies" / date_str / f"{self.params.theme_slug}_{time_str}"
        _ensure_dir(self.run_dir)

    # ===== Implementacja kroków =====
    def _load_excel(self) -> pd.DataFrame:
        df = pd.read_excel(self.params.excel_path)
        # Normalizacja kolumn
        if 'post_id' not in df.columns:
            df['post_id'] = df.get('id') if 'id' in df.columns else pd.RangeIndex(start=1, stop=len(df)+1, step=1)
        # Ujednolić typ klucza do string (spójny z danymi z LLM)
        try:
            df['post_id'] = df['post_id'].astype(str)
        except Exception:
            df['post_id'] = df['post_id'].map(lambda x: str(x))
        if 'content' not in df.columns:
            # Spróbuj wykryć prawdopodobną kolumnę tekstu
            candidates = [c for c in df.columns if str(c).lower() in ('text', 'content', 'post', 'message', 'body')]
            if candidates:
                df.rename(columns={candidates[0]: 'content'}, inplace=True)
            else:
                raise ValueError("W Excelu brakuje kolumny 'content'.")
        if self.params.max_posts and self.params.max_posts > 0:
            df = df.head(self.params.max_posts)
        return df

    def _step_1_summarize(self, df: pd.DataFrame, debug_dir: Optional[Path] = None) -> pd.DataFrame:
        rows = df.to_dict(orient='records')
        out: List[Dict[str, Any]] = []
        for bi, batch in enumerate(_chunks(rows, self.params.batch_size)):
            messages = _render_prompt_summarize(batch)
            txt = _chat(self.client, self.llm, messages)
            if debug_dir is not None:
                try:
                    (debug_dir / f"step1_batch_{bi:03d}_request.json").write_text(
                        json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8'
                    )
                    (debug_dir / f"step1_batch_{bi:03d}_response.txt").write_text(txt, encoding='utf-8')
                except Exception:
                    pass
            data = _parse_json_safe(txt)
            if not isinstance(data, list):
                data = []
            # Wymuś poprawne mapowanie po indeksie batcha i post_id
            for i, row in enumerate(batch):
                item = data[i] if i < len(data) and isinstance(data[i], dict) else {}
                out.append({
                    'post_id': str(row.get('post_id', row.get('id', i))),
                    'summary': _safe_text(item.get('summary')),
                    'quote': _safe_text(item.get('quote')),
                })
        m = pd.DataFrame(out)
        merged = df.merge(m[['post_id', 'summary', 'quote']], on='post_id', how='left')
        return merged

    def _step_2_suggest_main(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = df.to_dict(orient='records')
        sugg: List[Dict[str, Any]] = []
        for batch in _chunks(rows, self.params.batch_size):
            messages = _render_prompt_suggest_categories(self.params.theme_slug, batch)
            txt = _chat(self.client, self.llm, messages)
            data = _parse_json_safe(txt)
            if not isinstance(data, list):
                data = []
            for i, row in enumerate(batch):
                item = data[i] if i < len(data) and isinstance(data[i], dict) else {}
                sugg.append({
                    'post_id': str(row.get('post_id', row.get('id', i))),
                    'candidates': item.get('candidates', []),
                })
        ms = pd.DataFrame(sugg)
        # ujednolicenie listy kandydatów do CSV
        def _to_csv(x: Any) -> str:
            if isinstance(x, list):
                return ", ".join([_safe_text(i) for i in x if _safe_text(i)])
            return _safe_text(x)
        if 'candidates' in ms.columns:
            ms['main_candidates'] = ms['candidates'].apply(_to_csv)
        else:
            ms['main_candidates'] = ''
        merged = df.merge(ms[['post_id', 'main_candidates']], on='post_id', how='left')
        return merged

    def _step_3_consolidate_main(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        all_cands: List[str] = []
        for val in df.get('main_candidates', pd.Series([], dtype=str)).fillna(''):
            parts = [p.strip() for p in str(val).split(',') if str(p).strip()]
            all_cands.extend(parts)
        # Dedup lokalny, potem LLM konsolidacja
        uniq = sorted({c for c in all_cands if c})
        if not uniq:
            raise RuntimeError("Brak kandydatur kategorii głównych.")
        messages = _render_prompt_consolidate_main(uniq)
        txt = _chat(self.client, self.llm, messages)
        data = _parse_json_safe(txt)
        if not isinstance(data, list) or not data:
            raise RuntimeError("Model nie zwrócił głównych kategorii w oczekiwanym formacie.")
        return [
            {"name": _safe_text(d.get('name')), "desc": _safe_text(d.get('desc'))}
            for d in data if _safe_text(d.get('name'))
        ]

    def _sample_rows_for_category(self, df: pd.DataFrame, name: str, k: int = 40) -> List[Dict[str, Any]]:
        # Prosty sampling: bierz te posty, gdzie nazwa pojawia się w main_candidates
        mask = df.get('main_candidates', pd.Series([], dtype=str)).fillna('').str.contains(name, case=False, na=False)
        cand_df = df[mask]
        if len(cand_df) == 0:
            cand_df = df.sample(n=min(k, len(df)), random_state=42) if len(df) > 0 else df
        else:
            cand_df = cand_df.head(k)
        return cand_df[['summary', 'quote']].fillna('').to_dict(orient='records')

    def _step_4_5_build_hierarchy(self, df: pd.DataFrame, mains: List[Dict[str, str]]) -> Dict[str, Any]:
        tree: Dict[str, Any] = {"children": []}
        # Poziom 2 i 3
        for m in mains:
            m_name = m['name']
            examples = self._sample_rows_for_category(df, m_name, k=40)
            # Podkategorie (3–8)
            msg_sub = _render_prompt_induce_children("podkategorii", m_name, examples, 3, 8)
            txt_sub = _chat(self.client, self.llm, msg_sub)
            subs = _parse_json_safe(txt_sub)
            sub_nodes: List[Dict[str, Any]] = []
            if isinstance(subs, list):
                for s in subs:
                    s_name = _safe_text(s.get('name'))
                    if not s_name:
                        continue
                    # Pod‑podkategorie (2–8)
                    msg_subsub = _render_prompt_induce_children("pod‑podkategorii", s_name, examples, 2, 8)
                    txt_subsub = _chat(self.client, self.llm, msg_subsub)
                    subsubs = _parse_json_safe(txt_subsub)
                    subsub_nodes: List[Dict[str, Any]] = []
                    if isinstance(subsubs, list):
                        for ss in subsubs:
                            ss_name = _safe_text(ss.get('name'))
                            if not ss_name:
                                continue
                            subsub_nodes.append({
                                "name": ss_name,
                                "desc": _safe_text(ss.get('desc')),
                            })
                    sub_nodes.append({
                        "name": s_name,
                        "desc": _safe_text(s.get('desc')),
                        "children": subsub_nodes,
                    })
            tree["children"].append({
                "name": m_name,
                "desc": _safe_text(m.get('desc')),
                "children": sub_nodes,
            })
        return tree

    def _step_6_numbering(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Najpierw spróbuj numeracji lokalnej (bez LLM) – stabilna i szybka
        try:
            numbered = _local_numbering(tree)
            if isinstance(numbered, list) and numbered:
                return numbered
        except Exception:
            pass

        # Fallback do LLM tylko jeśli lokalna się nie powiodła
        messages = _render_prompt_assemble_numbering(tree)
        txt = _chat(self.client, self.llm, messages)
        data = _parse_json_safe(txt)
        if not isinstance(data, list):
            raise RuntimeError("Model nie zwrócił znumerowanej hierarchii jako listy.")
        return data

    def _step_7_assign(self, taxonomy_json: Any, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows = df[['post_id', 'summary', 'quote']].fillna('').to_dict(orient='records')
        assigns: List[Dict[str, Any]] = []
        for batch in _chunks(rows, self.params.batch_size):
            messages = _render_prompt_assign(taxonomy_json, batch)
            txt = _chat(self.client, self.llm, messages)
            data = _parse_json_safe(txt)
            if isinstance(data, list):
                assigns.extend(data)
        return assigns

    def _step_10_compile_prompt(self, taxonomy_json: Any) -> str:
        messages = _render_prompt_compile_system_message(taxonomy_json)
        txt = _chat(self.client, self.llm, messages)
        return txt.strip()

    # ===== Orkiestracja =====
    def build(self) -> Dict[str, Any]:
        df = self._load_excel()
        df = self._step_1_summarize(df)
        df = self._step_2_suggest_main(df)
        mains = self._step_3_consolidate_main(df)
        tree = self._step_4_5_build_hierarchy(df, mains)
        taxonomy_numbered = self._step_6_numbering(tree)
        assignments = self._step_7_assign(taxonomy_numbered, df)

        # Eksporty
        art_dir = self.run_dir
        _ensure_dir(art_dir)
        taxonomy_path = art_dir / "taxonomy.json"
        assignments_path = art_dir / "assignments.jsonl"
        excel_out_path = art_dir / f"labeled_{self.params.excel_path.name}"

        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_numbered, f, ensure_ascii=False, indent=2)

        with open(assignments_path, 'w', encoding='utf-8') as f:
            for a in assignments:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

        # Dołącz przypisania do Excela
        asg_df = pd.DataFrame(assignments)
        if 'post_id' in asg_df.columns:
            df_out = df.merge(asg_df[['post_id', 'path', 'confidence']], on='post_id', how='left')
        else:
            df_out = df
        with pd.ExcelWriter(excel_out_path, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='labeled')

        # Kompilacja promptu systemowego
        system_message = self._step_10_compile_prompt(taxonomy_numbered)
        prompt_path = art_dir / "prompt_system_message.md"
        prompt_path.write_text(system_message, encoding='utf-8')

        result = {
            'run_dir': str(art_dir.resolve()),
            'taxonomy_path': str(taxonomy_path.resolve()),
            'assignments_path': str(assignments_path.resolve()),
            'prompt_path': str(prompt_path.resolve()),
            'excel_out_path': str(excel_out_path.resolve()),
        }
        return result


def run_taxonomy_pipeline(excel_path: str, output_dir: str, theme_slug: str,
                          batch_size: int = 50, max_posts: Optional[int] = None) -> Dict[str, Any]:
    params = BuilderParams(
        excel_path=Path(excel_path),
        output_base_dir=Path(output_dir),
        theme_slug=theme_slug,
        batch_size=batch_size,
        max_posts=max_posts,
    )
    builder = TaxonomyPromptBuilder(params)
    return builder.build()


__all__ = [
    'BuilderParams',
    'LLMParams',
    'TaxonomyPromptBuilder',
    'run_taxonomy_pipeline',
]


