#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Klasyfikacja postów z Excela do zdefiniowanej taksonomii z użyciem OpenAI Batch API.

Funkcjonalności:
- Wczytanie Excela (kolumny: content, opcjonalnie post_id)
- Podział na batch'e po 10 postów i tworzenie jobów Batch API (chat.completions)
- Polling statusu, pobranie outputu, zapis wyników JSON dla batchy
- Dynamiczne rozszerzanie taksonomii o nowe podkategorie (X.n lub X.Y.n)
- Fuzja wszystkich wyników i merge z wejściowym Excelem (dodanie kolumn)

Wymagania:
- OPENAI_API_KEY w środowisku (.env) lub w analysis/config.py -> LLM_CONFIG['api_key']
- Model: gpt-5-mini (domyślnie)
"""

from __future__ import annotations

import json
import os
import random
import string
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from analysis.config import LLM_CONFIG  # type: ignore
except Exception:  # pragma: no cover
    LLM_CONFIG = {
        'provider': 'openrouter',
        'model': 'gpt-5-mini',
        'temperature': 0.5,
        'max_tokens': 800,
        'api_key': os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
        'base_url': os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        'default_headers': None,
    }


# ==== Parametry i pomocnicze ====

@dataclass
class BatchRunParams:
    excel_path: Path
    output_dir: Path
    batch_size: int = 10
    poll_interval_s: int = 10
    max_requests_per_batch_job: Optional[int] = None  # None = wszystkie w ramach 10
    model: str = str(LLM_CONFIG.get('model', 'gpt-5-mini'))
    temperature: float = float(LLM_CONFIG.get('temperature', 0.5))
    max_tokens: int = int(LLM_CONFIG.get('max_tokens', 800))


def _rand_suffix(n: int = 6) -> str:
    return ''.join(random.choices(string.hexdigits.lower(), k=n))


def _now_slug() -> Tuple[str, str]:
    dt = datetime.now()
    return dt.strftime('%Y%m%d'), dt.strftime('%H%M%S')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_str(x: Any) -> str:
    return (str(x) if x is not None else '').strip()


def _load_excel_posts(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    # Ustal post_id
    if 'post_id' not in df.columns:
        if 'id' in df.columns:
            df['post_id'] = df['id']
        else:
            df['post_id'] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
    try:
        df['post_id'] = df['post_id'].astype(str)
    except Exception:
        df['post_id'] = df['post_id'].map(lambda x: str(x))
    # Ustal content
    if 'content' not in df.columns:
        candidates = [c for c in df.columns if str(c).lower() in ('text', 'content', 'post', 'message', 'body')]
        if candidates:
            df.rename(columns={candidates[0]: 'content'}, inplace=True)
        else:
            raise ValueError("W Excelu brakuje kolumny 'content'.")
    return df[['post_id', 'content']].copy()


# ==== Taksonomia (2 poziomy: główny i podkategorie; dopuszczamy zagnieżdżenie 3-poziomowe jako sub_id X.Y.Z) ====

def _base_taxonomy() -> Dict[str, Any]:
    """Zwraca drzewo taksonomii w postaci:
    {
      '1': {'name': 'Orientacja ...', 'children': {'1.1': {'name': 'Pro-PiS'}, ...}},
      '2': {...},
    }
    """
    # Uwaga: to odwzorowanie listy dostarczonej przez użytkownika.
    t: Dict[str, Any] = {
        '1': {
            'name': 'Orientacja polityczna i autoidentyfikacja',
            'children': {
                '1.1': {'name': 'Pro-PiS'},
                '1.2': {'name': 'Anty-PiS'},
                '1.3': {'name': 'Pro-PO/KO'},
                '1.4': {'name': 'Anty-PO/KO'},
                '1.5': {'name': 'Pro-SLD/Lewica'},
                '1.6': {'name': 'Anty-SLD/Lewica'},
                '1.7': {'name': 'Pro-PSL'},
                '1.8': {'name': 'Anty-PSL'},
                '1.9': {'name': 'Pro-Konfederacja'},
                '1.10': {'name': 'Anty-Konfederacja'},
                '1.11': {'name': 'Pro-Kukiz’15'},
                '1.12': {'name': 'Anty-Kukiz’15'},
                '1.13': {'name': 'Pro-Prawica RP'},
                '1.14': {'name': 'Anty-Prawica RP'},
                '1.15': {'name': 'Pro-Nowoczesna'},
                '1.16': {'name': 'Anty-Nowoczesna'},
                '1.17': {'name': 'Pro-Hołownia/Polska2050'},
                '1.18': {'name': 'Anty-Hołownia/Polska2050'},
                '1.19': {'name': 'Deklaracja „mniejsze zło”'},
                '1.20': {'name': 'Deklaracja „nie głosuję/bojkot”'},
                '1.21': {'name': 'Głosowanie poza duopolem/spoza Sejmu'},
                '1.22': {'name': 'Niezdecydowanie / ambiwalencja'},
            },
        },
        '2': {
            'name': 'Strategie wyborcze i zachowania',
            'children': {
                '2.1': {'name': 'Głosowanie taktyczne'},
                '2.2': {'name': 'Z sumieniem vs „sondażowe”'},
                '2.3': {'name': 'Wpływ sondaży'},
                '2.4': {'name': 'JOW / reformy ordynacji'},
                '2.5': {'name': '„Stracony głos” – narracje'},
                '2.6': {'name': 'Frekwencja / mobilizacja'},
                '2.7': {'name': '„Mniejsze zło” w II turze'},
                '2.8': {'name': 'Głos nieważny / rezygnacja'},
                '2.9': {'name': 'Koalicje taktyczne oddolne'},
                '2.10': {'name': 'Debaty – bojkot/uczestnictwo'},
            },
        },
        '3': {
            'name': 'Spory programowe i ideowe (tematy)',
            'children': {
                '3.1': {
                    'name': 'Gospodarka i podatki',
                    'children': {
                        '3.1.1': {'name': 'Podatek liniowy vs progresja'},
                        '3.1.2': {'name': '500+ i polityka socjalna'},
                        '3.1.3': {'name': 'Interwencjonizm vs leseferyzm'},
                        '3.1.4': {'name': 'Dług/deficyt socjalizm vs liberalizm'},
                        '3.1.5': {'name': 'Prywatyzacje / własność państwowa'},
                    },
                },
                '3.2': {
                    'name': 'Światopogląd i bioetyka',
                    'children': {
                        '3.2.1': {'name': 'Aborcja – regulacje'},
                        '3.2.2': {'name': 'In vitro'},
                        '3.2.3': {'name': 'Związki partnerskie / małżeństwa jednopłciowe / adopcje'},
                        '3.2.4': {'name': 'Edukacja seksualna / „ideologia LGBT”'},
                    },
                },
                '3.3': {
                    'name': 'Kościół i religia w polityce',
                    'children': {
                        '3.3.1': {'name': 'Rola Kościoła / mediów katolickich'},
                        '3.3.2': {'name': 'Krzyż / symbole religijne'},
                        '3.3.3': {'name': '„Partia dla katolików” / głos katolika'},
                    },
                },
                '3.4': {
                    'name': 'Państwo prawa i instytucje',
                    'children': {
                        '3.4.1': {'name': 'TK / sądy / prokuratura'},
                        '3.4.2': {'name': 'Służby i lustracja/dekomunizacja'},
                        '3.4.3': {'name': 'Media publiczne – stronniczość'},
                        '3.4.4': {'name': 'Spółki SP / nominacje polityczne'},
                    },
                },
                '3.5': {
                    'name': 'Polityka zagraniczna i bezpieczeństwo',
                    'children': {
                        '3.5.1': {'name': 'UE / KPO / Zielony Ład'},
                        '3.5.2': {'name': 'NATO/USA – obecność wojsk'},
                        '3.5.3': {'name': 'Rosja/Ukraina – relacje'},
                    },
                },
                '3.6': {
                    'name': 'Polityka energetyczno‑klimatyczna',
                    'children': {
                        '3.6.1': {'name': 'Węgiel i kopalnie'},
                        '3.6.2': {'name': 'Zielony Ład / transformacja'},
                        '3.6.3': {'name': 'Import odpadów / „ekologiczna prawica”'},
                    },
                },
                '3.7': {
                    'name': 'Polityka społeczna i demografia',
                    'children': {
                        '3.7.1': {'name': 'Rodzina, demografia, świadczenia'},
                        '3.7.2': {'name': 'Migracja / integracja cudzoziemców'},
                        '3.7.3': {'name': 'Zdrowie publiczne (pandemia)'},
                    },
                },
                '3.8': {
                    'name': 'Media i informacja',
                    'children': {
                        '3.8.1': {'name': '„Propaganda”, „fałszywe newsy”'},
                        '3.8.2': {'name': 'Prywatne vs publiczne media'},
                        '3.8.3': {'name': 'Cenzura / „cisza wyborcza”'},
                    },
                },
            },
        },
        '4': {
            'name': 'Aktorzy, osoby i kult jednostki',
            'children': {
                '4.1': {'name': 'Liderzy PiS'},
                '4.2': {'name': 'Liderzy PO/KO'},
                '4.3': {'name': 'Lewica'},
                '4.4': {'name': 'PSL'},
                '4.5': {'name': 'Konfederacja'},
                '4.6': {'name': 'Inni (Kukiz, Hołownia,...)'},
                '4.7': {'name': 'Dziennikarze/medialni'},
                '4.8': {'name': 'Autorytety religijne'},
                '4.9': {'name': 'Historyczne postacie i porównania'},
            },
        },
        '5': {
            'name': 'Koalicje, alianse i układy',
            'children': {
                '5.1': {'name': 'Realne i spekulatywne koalicje'},
                '5.2': {'name': '„POPIS” – krytyka duopolu'},
                '5.3': {'name': 'Koalicje medialne'},
                '5.4': {'name': '„Układ”, „Magdalenka”, „Okrągły Stół”'},
                '5.5': {'name': 'Transfery posłów / zdrady'},
            },
        },
        '6': {
            'name': 'Retoryka i styl wypowiedzi',
            'children': {
                '6.1': {'name': 'Ad personam / inwektywy / ironia'},
                '6.2': {'name': 'Etykietowanie'},
                '6.3': {'name': 'Teorie spiskowe'},
                '6.4': {'name': 'Straszenie przeciwnikiem'},
                '6.5': {'name': 'Apokaliptyczne diagnozy'},
                '6.6': {'name': 'Apel religijny/moralny'},
                '6.7': {'name': 'Sarkazm/żart/hiperbola'},
                '6.8': {'name': 'Własne świadectwo i anegdoty'},
                '6.9': {'name': 'Odwołanie do sondaży jako argument'},
                '6.10': {'name': 'Kontrfaktyczne porównania historyczne'},
            },
        },
        '7': {
            'name': 'Procedury wyborcze i prawne',
            'children': {
                '7.1': {'name': 'Cisza wyborcza, agitacja'},
                '7.2': {'name': 'Zarzuty o nieuczciwość wyborów'},
                '7.3': {'name': 'PKW / ordynacja / progi'},
                '7.4': {'name': 'II tura – przepływy elektoratów'},
                '7.5': {'name': 'Rola debat / kalendarz / terminy'},
            },
        },
        '8': {
            'name': 'Media i metakomentarz o debacie publicznej',
            'children': {
                '8.1': {'name': '„Polaryzacja” sceny politycznej'},
                '8.2': {'name': '„Zabetonowanie” duopolu'},
                '8.3': {'name': 'Rola celebrytów/artystów w kampanii'},
                '8.4': {'name': 'Bańki informacyjne i echo chambers'},
                '8.5': {'name': 'Sondażokracja i „ustawianie” wyborów'},
            },
        },
        '9': {
            'name': 'Wątki historyczno-symboliczne',
            'children': {
                '9.1': {'name': 'Smoleńsk – pamięć, spory'},
                '9.2': {'name': 'Lustracja/dekomunizacja/IV RP'},
                '9.3': {'name': 'Symbole (krzyż na Krakowskim Przedmieściu)'},
                '9.4': {'name': '„Nocna zmiana” / filmy i narracje'},
            },
        },
        '10': {
            'name': 'Postawy wobec UE i integracji europejskiej',
            'children': {
                '10.1': {'name': 'Euroentuzjazm / proeuropejskość'},
                '10.2': {'name': 'Eurorealizm / sceptycyzm'},
                '10.3': {'name': '„Polexit” – lęk/zarzut/polemika'},
                '10.4': {'name': 'KPO / relacje instytucjonalne'},
                '10.5': {'name': 'Zielony Ład – postawy'},
            },
        },
        '11': {
            'name': 'Kategorie jakościowe (emocje i ton)',
            'children': {
                '11.1': {'name': 'Ton pojednawczy / „ponad podziałami”'},
                '11.2': {'name': 'Ton konfrontacyjny / polaryzujący'},
                '11.3': {'name': 'Ton moralizatorski / normatywny'},
                '11.4': {'name': 'Ton pragmatyczny / technokratyczny'},
                '11.5': {'name': 'Ton cyniczny / zniechęcony'},
            },
        },
        '12': {
            'name': 'Metryki oparte na treści (analityka)',
            'children': {
                '12.1': {'name': 'Stosunek do sondaży'},
                '12.2': {'name': 'Kandydat optymalny (taktycznie/ideowo)'},
                '12.3': {'name': 'Indeks polaryzacji'},
                '12.4': {'name': 'Indeks moralizacji'},
                '12.5': {'name': 'Indeks spiskowy'},
                '12.6': {'name': 'Indeks programowy'},
                '12.7': {'name': 'Indeks historyzacji'},
                '12.8': {'name': 'Indeks medialny'},
            },
        },
        '13': {
            'name': 'Specjalne etykiety tematyczne (przekrojowe)',
            'children': {
                '13.1': {'name': 'Antypopulizm / antyelitaryzm'},
                '13.2': {'name': 'Antyklerykalizm / klerykalizm'},
                '13.3': {'name': '„Socjalizm” vs „liberalizm” – etykiety'},
                '13.4': {'name': '„Agentura rosyjska/niemiecka” – oskarżenia'},
                '13.5': {'name': '„Mniejsze zło” – uzasadnienia'},
                '13.6': {'name': '„Sondaż jako performowanie rzeczywistości”'},
                '13.7': {'name': '„Media jako aktor polityczny”'},
            },
        },
        '14': {
            'name': 'Kanał i kontekst wypowiedzi',
            'children': {
                '14.1': {'name': 'Forum katolickie'},
                '14.2': {'name': 'Forum świeckie/ogólnopolityczne'},
                '14.3': {'name': 'Odwołania do konkretnych wątków/portali'},
                '14.4': {'name': 'Cytaty z debat/wywiadów/eksposé'},
            },
        },
    }
    return t


def _taxonomy_to_text(tax: Dict[str, Any]) -> str:
    lines: List[str] = []
    for main_id in sorted(tax.keys(), key=lambda x: [int(p) for p in x.split('.')]):
        main = tax[main_id]
        lines.append(f"{main_id}. {main.get('name','')}")
        children = main.get('children', {}) or {}
        # wypisz rekursywnie do 3 poziomu
        for sub_id in sorted(children.keys(), key=lambda x: [int(p) for p in x.split('.')]):
            sub = children[sub_id]
            lines.append(f"- {sub_id}. {sub.get('name','')}")
            sub_children = sub.get('children', {}) or {}
            for subsub_id in sorted(sub_children.keys(), key=lambda x: [int(p) for p in x.split('.')]):
                subsub = sub_children[subsub_id]
                lines.append(f"  - {subsub_id}. {subsub.get('name','')}")
    return "\n".join(lines)


def _add_new_subcategories(tax: Dict[str, Any], proposals: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Dodaje nowe podkategorie do taksonomii. proposals: [{'parent_id':'1','name':'...'}].
    Zwraca listę faktycznie dodanych z ich nadanym id: [{'id':'1.23','name':'...','parent_id':'1'}].
    Obsługuje także rodziców 2‑poziomowych typu '3.1'.
    """
    added: List[Dict[str, str]] = []
    for p in proposals or []:
        parent_id = _safe_str(p.get('parent_id'))
        name = _safe_str(p.get('name'))
        if not parent_id or not name:
            continue
        # Rozpoznaj poziom rodzica
        parts = parent_id.split('.')
        if len(parts) == 1:
            # rodzic główny (np. '1')
            main = tax.get(parent_id)
            if not isinstance(main, dict):
                continue
            ch = main.setdefault('children', {})
            # wyznacz następny indeks
            existing = [cid for cid in ch.keys() if cid.startswith(parent_id + '.')]
            next_idx = 1
            if existing:
                try:
                    next_idx = max(int(e.split('.')[1]) for e in existing) + 1
                except Exception:
                    next_idx = len(existing) + 1
            new_id = f"{parent_id}.{next_idx}"
            if new_id in ch:
                # awaryjnie inkrementuj do unikalności
                k = next_idx
                while f"{parent_id}.{k}" in ch:
                    k += 1
                new_id = f"{parent_id}.{k}"
            ch[new_id] = {'name': name}
            added.append({'id': new_id, 'name': name, 'parent_id': parent_id})
        elif len(parts) == 2:
            # rodzic drugiego poziomu (np. '3.1')
            main = tax.get(parts[0])
            if not isinstance(main, dict):
                continue
            sub = (main.get('children') or {}).get(parent_id)
            if not isinstance(sub, dict):
                continue
            ch2 = sub.setdefault('children', {})
            existing = [cid for cid in ch2.keys() if cid.startswith(parent_id + '.')]
            next_idx = 1
            if existing:
                try:
                    next_idx = max(int(e.split('.')[2]) for e in existing) + 1
                except Exception:
                    next_idx = len(existing) + 1
            new_id = f"{parent_id}.{next_idx}"
            if new_id in ch2:
                k = next_idx
                while f"{parent_id}.{k}" in ch2:
                    k += 1
                new_id = f"{parent_id}.{k}"
            ch2[new_id] = {'name': name}
            added.append({'id': new_id, 'name': name, 'parent_id': parent_id})
        # ignoruj głębsze poziomy (max 3)
    return added


def _render_system_prompt(tax: Dict[str, Any]) -> str:
    return (
        "Zadanie: Przypisz post do 1–3 kategorii (główna + podkategoria) z katalogu. "
        "Używaj istniejących podkategorii. TYLKO gdy żadna nie pasuje, zaproponuj NOWĄ podkategorię.\n"
        "W razie nowej podkategorii wskaż rodzica przez parent_id (np. '1' lub '3.1') i krótką nazwę.\n"
        "Zwróć WYŁĄCZNIE JSON:")


def _render_user_prompt(post_id: str, content: str, tax: Dict[str, Any]) -> str:
    schema = {
        "post_id": post_id,
        "choices": [
            {
                "main_id": "X",
                "sub_id": "X.Y lub X.Y.Z",
                "main_label": "...",
                "sub_label": "..."
            }
        ],
        "summary": "1 zdanie (do 3 krótkich), bez wstępu typu 'Autor uważa...'.",
        "new_subcategories": [
            {"parent_id": "X lub X.Y", "name": "tylko jeśli naprawdę brak pasującej"}
        ]
    }
    catalog = _taxonomy_to_text(tax)
    return (
        f"Katalog kategorii:\n{catalog}\n\n"
        f"Post (id={post_id}):\n" + content.strip() + "\n\n"
        f"Format JSON:\n" + json.dumps(schema, ensure_ascii=False)
    )


def _is_openrouter() -> bool:
    prov = _safe_str(LLM_CONFIG.get('provider')) if isinstance(LLM_CONFIG, dict) else ''
    base = _safe_str(LLM_CONFIG.get('base_url')) if isinstance(LLM_CONFIG, dict) else ''
    return prov.lower() == 'openrouter' or 'openrouter.ai' in base.lower()


def _get_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("Brak biblioteki openai. Zainstaluj 'openai'.")
    # Klucz: preferuj LLM_CONFIG['api_key'] (może wskazywać OPENROUTER_API_KEY)
    cfg_key = _safe_str(LLM_CONFIG.get('api_key')) if isinstance(LLM_CONFIG, dict) else ''
    env_or = os.environ.get('OPENROUTER_API_KEY', '')
    env_oa = os.environ.get('OPENAI_API_KEY', '')
    api_key = cfg_key or env_or or env_oa
    if not api_key:
        raise RuntimeError("Brak klucza API (OPENROUTER_API_KEY/OPENAI_API_KEY).")
    kwargs: Dict[str, Any] = {'api_key': api_key}
    # Dla OpenRouter ustaw base_url i opcjonalne nagłówki
    if _is_openrouter():
        base = _safe_str(LLM_CONFIG.get('base_url'))
        if base:
            kwargs['base_url'] = base
        headers = LLM_CONFIG.get('default_headers') if isinstance(LLM_CONFIG, dict) else None
        if isinstance(headers, dict) and headers:
            kwargs['default_headers'] = headers
    return OpenAI(**kwargs)


def _chunk_iter(lst: List[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        yield lst
        return
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _parse_json_safe(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # Spróbuj wyciąć blok JSON
        s1, s2 = text.find('{'), text.find('[')
        starts = [x for x in (s1, s2) if x != -1]
        if not starts:
            return None
        s = min(starts)
        e1, e2 = text.rfind('}'), text.rfind(']')
        e = max(e1, e2)
        if e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
        return None


def _build_batch_jsonl_lines(rows: List[Dict[str, str]], tax: Dict[str, Any], params: BatchRunParams) -> List[str]:
    lines: List[str] = []
    for r in rows:
        pid = _safe_str(r.get('post_id'))
        content = _safe_str(r.get('content'))
        system_msg = _render_system_prompt(tax)
        user_msg = _render_user_prompt(pid, content, tax)
        body = {
            "model": params.model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        line = {
            "custom_id": f"post-{pid}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        lines.append(json.dumps(line, ensure_ascii=False))
    return lines


def _upload_batch_input(client: Any, jsonl_path: Path) -> Any:
    with open(jsonl_path, 'rb') as f:
        return client.files.create(file=f, purpose="batch")


def _create_batch_job(client: Any, file_id: str) -> Any:
    return client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )


def _retrieve_file_text(client: Any, file_id: str) -> str:
    resp = client.files.content(file_id)
    # OpenAI Python SDK 1.x zwraca obiekt z metodami read()/iter_bytes()/stream_to_file()
    try:
        if hasattr(resp, 'text') and isinstance(getattr(resp, 'text'), str):
            return resp.text  # type: ignore[attr-defined]
        if hasattr(resp, 'read'):
            data = resp.read()
            try:
                return data.decode('utf-8')
            except Exception:
                return data.decode('utf-8', errors='ignore')
        if hasattr(resp, 'content'):
            data = resp.content  # type: ignore[attr-defined]
            try:
                return data.decode('utf-8')
            except Exception:
                return str(data)
    except Exception:
        pass
    # Fallback – próbuj str()
    try:
        return str(resp)
    except Exception:
        return ""


def _poll_until_complete(client: Any, batch_id: str, interval_s: int, state_path: Path) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, 'status', None) or batch.get('status')
        # zapisuj stan
        try:
            _write_json(state_path, {
                'batch_id': getattr(batch, 'id', None) or batch.get('id'),
                'status': status,
                'request_counts': getattr(batch, 'request_counts', None) or batch.get('request_counts'),
                'timestamps': {
                    'created_at': getattr(batch, 'created_at', None) or batch.get('created_at'),
                    'in_progress_at': getattr(batch, 'in_progress_at', None) or batch.get('in_progress_at'),
                    'completed_at': getattr(batch, 'completed_at', None) or batch.get('completed_at'),
                    'failed_at': getattr(batch, 'failed_at', None) or batch.get('failed_at'),
                }
            })
        except Exception:
            pass
        if status in ("completed", "failed", "cancelled", "expired"):
            return batch
        time.sleep(max(1, interval_s))


def _chat_direct(client: Any, params: BatchRunParams, system_msg: str, user_msg: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=params.model,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    body = {
        'id': getattr(resp, 'id', None),
        'model': getattr(resp, 'model', None),
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': _safe_str(getattr(resp.choices[0].message, 'content', '')) if getattr(resp, 'choices', None) else ''
                },
                'finish_reason': getattr(resp.choices[0], 'finish_reason', None) if getattr(resp, 'choices', None) else None,
            }
        ] if getattr(resp, 'choices', None) else [],
    }
    return body


def run_batch_classification(
    excel_path: str = "data/topics/results/20250821/M/ALL/185832/examples/topic_2_pi_pis_sld.xlsx",
    batch_size: int = 10,
    poll_interval_s: int = 10,
) -> Dict[str, Any]:
    """Główna funkcja orkiestracji klasyfikacji.
    Uwaga: dla OpenRouter wykonywany jest tryb bez Batch API (bez uploadu/pollingu),
    a dla OpenAI używany jest Batch API.
    """
    xlsx = Path(excel_path)
    if not xlsx.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {xlsx}")

    df = _load_excel_posts(xlsx)
    records = df.to_dict(orient='records')

    # Katalog wyjściowy sesji
    date_str, time_str = _now_slug()
    run_dir = Path("data/topics/results") / f"llm_batch_{date_str}_{time_str}_{_rand_suffix()}"
    _ensure_dir(run_dir)

    # Stan: taksonomia
    taxonomy_path = run_dir / "taxonomy.json"
    taxonomy: Dict[str, Any]
    taxonomy = _base_taxonomy()
    _write_json(taxonomy_path, taxonomy)

    # Klient OpenAI/OpenRouter
    client = _get_openai_client()

    all_outputs: List[Dict[str, Any]] = []

    use_batch_api = not _is_openrouter()

    for batch_index, chunk in enumerate(_chunk_iter(records, batch_size)):
        batch_results: List[Dict[str, Any]] = []
        if use_batch_api:
            # === Tryb OpenAI Batch API ===
            lines = _build_batch_jsonl_lines(chunk, taxonomy, BatchRunParams(
                excel_path=xlsx,
                output_dir=run_dir,
                batch_size=batch_size,
                poll_interval_s=poll_interval_s,
            ))
            in_path = run_dir / f"batch_{batch_index:04d}_input.jsonl"
            in_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

            uploaded = _upload_batch_input(client, in_path)
            input_file_id = getattr(uploaded, 'id', None) or uploaded.get('id')
            job = _create_batch_job(client, input_file_id)
            batch_id = getattr(job, 'id', None) or job.get('id')

            state_path = run_dir / f"batch_{batch_index:04d}_state.json"
            final = _poll_until_complete(client, batch_id, poll_interval_s, state_path)

            output_file_id = getattr(final, 'output_file_id', None) or final.get('output_file_id')
            error_file_id = getattr(final, 'error_file_id', None) or final.get('error_file_id')

            if output_file_id:
                txt = _retrieve_file_text(client, output_file_id)
                out_lines = [l for l in txt.splitlines() if l.strip()]
                for l in out_lines:
                    try:
                        obj = json.loads(l)
                    except Exception:
                        continue
                    custom_id = obj.get('custom_id')
                    body = (((obj.get('response') or {}).get('body') or {}))
                    content = ''
                    try:
                        choices = (body.get('choices') or [])
                        if choices and choices[0].get('message'):
                            content = _safe_str(choices[0]['message'].get('content'))
                    except Exception:
                        content = ''
                    parsed = _parse_json_safe(content) if content else None
                    batch_results.append({
                        'custom_id': custom_id,
                        'raw': body,
                        'parsed': parsed,
                    })
            if error_file_id:
                try:
                    err_txt = _retrieve_file_text(client, error_file_id)
                    (run_dir / f"batch_{batch_index:04d}_errors.jsonl").write_text(err_txt, encoding='utf-8')
                except Exception:
                    pass
        else:
            # === Tryb OpenRouter (bez Batch API) ===
            params = BatchRunParams(
                excel_path=xlsx,
                output_dir=run_dir,
                batch_size=batch_size,
                poll_interval_s=poll_interval_s,
            )
            for r in chunk:
                pid = _safe_str(r.get('post_id'))
                content = _safe_str(r.get('content'))
                system_msg = _render_system_prompt(taxonomy)
                user_msg = _render_user_prompt(pid, content, taxonomy)
                try:
                    body = _chat_direct(client, params, system_msg, user_msg)
                    content_txt = ''
                    try:
                        if body.get('choices') and body['choices'][0].get('message'):
                            content_txt = _safe_str(body['choices'][0]['message'].get('content'))
                    except Exception:
                        content_txt = ''
                    parsed = _parse_json_safe(content_txt) if content_txt else None
                except Exception as exc:
                    body = {'error': str(exc)}
                    parsed = None
                batch_results.append({
                    'custom_id': f'post-{pid}',
                    'raw': body,
                    'parsed': parsed,
                })
                # Krótki sleep, by nie zalać API
                time.sleep(0.1)

        # Zapisz wyniki batcha
        out_path = run_dir / f"batch_{batch_index:04d}_results.json"
        _write_json(out_path, batch_results)

        # Uaktualnij taksonomię jeśli są propozycje
        proposals: List[Dict[str, str]] = []
        for item in batch_results:
            p = item.get('parsed')
            if isinstance(p, dict) and isinstance(p.get('new_subcategories'), list):
                for pr in p['new_subcategories']:
                    pr_parent = _safe_str(pr.get('parent_id'))
                    pr_name = _safe_str(pr.get('name'))
                    if pr_parent and pr_name:
                        proposals.append({'parent_id': pr_parent, 'name': pr_name})
        added = _add_new_subcategories(taxonomy, proposals)
        if added:
            _write_json(run_dir / f"batch_{batch_index:04d}_new_subcategories.json", added)
            _write_json(taxonomy_path, taxonomy)

        # Agreguj do całości
        for item in batch_results:
            p = item.get('parsed')
            if isinstance(p, dict):
                all_outputs.append(p)

    # Zapisz całość
    combined_path = run_dir / "combined_results.json"
    _write_json(combined_path, all_outputs)

    # Merge do Excela
    # Normalizuj choices do kolumn: categories (CSV), summary
    rows_for_merge: List[Dict[str, Any]] = []
    for p in all_outputs:
        if not isinstance(p, dict):
            continue
        pid = _safe_str(p.get('post_id'))
        summary = _safe_str(p.get('summary'))
        cats = []
        for ch in p.get('choices') or []:
            try:
                mid = _safe_str(ch.get('main_id'))
                sid = _safe_str(ch.get('sub_id'))
                ml = _safe_str(ch.get('main_label'))
                sl = _safe_str(ch.get('sub_label'))
                cats.append(f"{mid}|{sid}|{ml}|{sl}")
            except Exception:
                continue
        rows_for_merge.append({
            'post_id': pid,
            'llm_summary': summary,
            'llm_categories': "; ".join(cats),
        })
    df_llm = pd.DataFrame(rows_for_merge)
    df_in = pd.read_excel(xlsx)
    if 'post_id' not in df_in.columns:
        if 'id' in df_in.columns:
            df_in['post_id'] = df_in['id']
        else:
            df_in['post_id'] = pd.RangeIndex(start=1, stop=len(df_in) + 1, step=1)
    try:
        df_in['post_id'] = df_in['post_id'].astype(str)
    except Exception:
        df_in['post_id'] = df_in['post_id'].map(lambda x: str(x))
    df_out = df_in.merge(df_llm, on='post_id', how='left')
    excel_out = run_dir / f"labeled_{xlsx.name}"
    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='labeled')

    # Podsumowanie
    result = {
        'run_dir': str(run_dir.resolve()),
        'taxonomy_path': str(taxonomy_path.resolve()),
        'combined_path': str(combined_path.resolve()),
        'excel_out_path': str(excel_out.resolve()),
        'total_posts': len(records),
        'batches': (len(records) + batch_size - 1) // batch_size,
    }
    (run_dir / 'state.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


__all__ = [
    'BatchRunParams',
    'run_batch_classification',
]


