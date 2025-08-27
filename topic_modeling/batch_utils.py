#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pomocnicze funkcje dla Batch API (OpenAI): klient i fallback klucza API.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from analysis.config import LLM_CONFIG  # type: ignore
except Exception:  # pragma: no cover
    LLM_CONFIG = {
        'model': 'gpt-4o-mini',
        'temperature': 0.7,
        'max_tokens': 1000,
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
    }


def get_openai_client() -> any:
    if OpenAI is None:
        raise RuntimeError("Biblioteka openai nie jest zainstalowana. Dodaj 'openai' do zależności.")
    cfg_key = str(LLM_CONFIG.get('api_key', '')).strip() if isinstance(LLM_CONFIG, dict) else ''
    env_key = os.environ.get('OPENAI_API_KEY', '').strip()
    api_key = cfg_key or env_key
    if not api_key:
        raise RuntimeError("Brak klucza API: ustaw analysis/config.py LLM_CONFIG['api_key'] lub OPENAI_API_KEY w env.")
    return OpenAI(api_key=api_key)


def get_default_model() -> str:
    return str(LLM_CONFIG.get('model', 'gpt-4o-mini'))


def truncate_text(text: Optional[str], max_len: int) -> str:
    s = str(text or '')
    if max_len <= 0:
        return s
    return s[:max_len]


