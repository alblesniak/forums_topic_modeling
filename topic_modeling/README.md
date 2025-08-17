# Modelowanie Tematyczne z Top2Vec

Ten moduł zawiera skrypty do analizy tematycznej danych z forów internetowych przy użyciu biblioteki Top2Vec.

## Opis

Skrypt analizuje dane z forów `radio_katolik` i `dolina_modlitwy` z bazy danych `analysis_forums.db`, wykonując modelowanie tematyczne osobno dla mężczyzn i kobiet.

## Parametry Analizy

- **Długość postów**: 25-1000 tokenów
- **Maksymalnie 100 postów od jednego użytkownika**
- **Równa liczba postów od kobiet i mężczyzn**
- **Minimalna liczba wystąpień słowa**: 5

## Struktura Katalogów

```
topic_modeling/
├── models/                    # Zapisane modele Top2Vec
│   ├── radio_katolik/
│   │   ├── male/
│   │   └── female/
│   └── dolina_modlitwy/
│       ├── male/
│       └── female/
├── results/                   # Wyniki analizy
│   ├── radio_katolik/
│   │   ├── male/
│   │   └── female/
│   └── dolina_modlitwy/
│       ├── male/
│       └── female/
├── logs/                      # Logi analizy
├── data/                      # Dane pośrednie
├── topic_modeling_script.py   # Główny skrypt analizy
├── run_analysis.py            # Skrypt pomocniczy
├── config.py                  # Konfiguracja
└── requirements.txt           # Wymagane biblioteki
```

## Instalacja

1. Zainstaluj wymagane biblioteki:

```bash
pip install -r requirements.txt
```

2. Zainstaluj bibliotekę Top2Vec:

```bash
pip install git+https://github.com/alblesniak/Top2Vec.git
```

## Użycie

### Uruchomienie wszystkich analiz

```bash
cd analysis/topic_modeling
python run_analysis.py
```

### Analiza konkretnego forum i płci

```bash
# Tylko radio_katolik - mężczyźni
python run_analysis.py --forum radio_katolik --gender male

# Tylko dolina_modlitwy - kobiety
python run_analysis.py --forum dolina_modlitwy --gender female
```

### Sprawdzenie zależności

```bash
python run_analysis.py --check-only
```

## Wyniki Analizy

Dla każdej kombinacji forum-płeć generowane są:

### 1. Model Top2Vec

- Zapisany w katalogu `models/{forum}/{gender}/`
- Może być później wczytany do dalszej analizy

### 2. Analiza Tematów

- Plik JSON z wynikami analizy
- Liczba tematów, słowa kluczowe, wyniki dokumentów
- Hierarchia tematów

### 3. Wizualizacje

- **Chmury słów** dla każdego tematu
- **Wykresy rozkładu** dokumentów w tematach
- **Wykresy wyników** dokumentów
- **Interaktywna mapa** tematów (HTML)

### 4. Logi

- Szczegółowe logi procesu analizy
- Informacje o błędach i ostrzeżeniach

## Konfiguracja

Parametry analizy można zmodyfikować w pliku `config.py`:

- `min_tokens`, `max_tokens` - długość postów
- `max_posts_per_user` - limit postów na użytkownika
- `min_word_count` - minimalna liczba wystąpień słowa
- `topic_merge_delta`, `topic_merge_threshold` - parametry łączenia tematów

## Przykład Wyników

```json
{
  "forum_name": "radio_katolik",
  "gender": "male",
  "num_topics": 15,
  "topics": [
    {
      "topic_id": 0,
      "words": ["modlitwa", "Bóg", "wiara", "kościół", "sakrament"],
      "word_scores": [0.95, 0.89, 0.87, 0.82, 0.79],
      "num_documents": 245,
      "avg_doc_score": 0.83
    }
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

## Rozwiązywanie Problemów

### Błąd: "Brak danych dla forum"

- Sprawdź czy baza danych zawiera dane dla danego forum
- Sprawdź czy istnieją posty z określoną płcią

### Błąd: "Za mało dokumentów do analizy"

- Zmniejsz parametry filtrowania w `config.py`
- Sprawdź czy dane spełniają kryteria długości

### Błąd: "Błąd podczas trenowania modelu"

- Sprawdź czy wszystkie biblioteki są zainstalowane
- Sprawdź logi w katalogu `logs/`

## Wymagania Systemowe

- Python 3.8+
- Minimum 8GB RAM (dla dużych baz danych)
- Dysk z wolną przestrzenią na modele i wyniki

## Licencja

Ten kod jest częścią projektu forums_scraper i podlega tym samym warunkom licencyjnym.
