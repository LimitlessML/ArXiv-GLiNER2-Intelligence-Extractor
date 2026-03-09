# ArXiv Intelligence Extractor

**Zero-shot NER on any ArXiv paper using GLiNER2 — extract models, datasets, metrics, authors and more in seconds.**

> Built with [GLiNER2](https://arxiv.org/abs/2507.18546) (fastino-ai, July 2025) — a DeBERTa-v3-large encoder fine-tuned for zero-shot named entity recognition. No task-specific training needed.

---

## What it does

Paste any ArXiv ID (e.g. `2507.18546`) and the app:

1. **Fetches** the full paper — HTML first, PDF fallback
2. **Chunks** the text (512 words, 50-word overlap) to stay within model context
3. **Extracts** 7 entity types using GLiNER2 zero-shot NER
4. **Visualizes** a knowledge graph, entity fingerprint radar, and an intelligence summary
5. **Links** entities to their [Papers With Code](https://paperswithcode.com) pages
6. **Compares** two papers side by side — shared entities, unique contributions, radar overlays

---

## Entity types extracted

| Label | Examples |
|---|---|
| ML model | GPT-4, LLaMA-3, DeBERTa-v3 |
| Dataset | ImageNet, SQuAD, MMLU |
| Metric | BLEU, F1, perplexity |
| Institution | Google DeepMind, MIT, Hugging Face |
| Method or technique | LoRA, RLHF, chain-of-thought |
| Benchmark | GLUE, BIG-bench, HELM |
| Author | Vaswani, LeCun, Bengio |

---

## Features

- **Zero-shot** — works on any paper without retraining
- **HTML + PDF** — scrapes structured HTML when available, falls back to PDF (pymupdf)
- **Key sections only** — filters introduction, methods, results, conclusion, experiments, abstract for speed
- **Knowledge graph** — entity co-occurrence network, colored by type, built with networkx + plotly
- **Radar chart** — paper "fingerprint" showing entity distribution across categories
- **Intelligence summary** — auto-generated narrative from extracted entities (no LLM required)
- **Papers With Code linking** — clickable URLs for models, datasets, methods
- **Multi-paper comparison** — shared vs unique entities, side-by-side fingerprints
- **CSV export** — download all entities with PWC links

---

## Architecture

```
app.py                    ← Streamlit UI (Single paper + Compare two papers)
extractor/
  fetcher.py              ← fetch_paper() — HTML scraping → PDF fallback → chunking
  ner.py                  ← GLiNER2 extraction, deduplication, key-section filtering
  visualizer.py           ← Knowledge graph, radar chart, intelligence summary
  linker.py               ← Papers With Code API entity linking
```

**Pipeline:**

```
ArXiv ID
   └─ fetch_paper()
        ├─ arxiv.org/html/{id}  →  BeautifulSoup → sections dict
        └─ arxiv.org PDF        →  pymupdf → single "full text" section
             └─ chunk_text(512w, 50w overlap)
                  └─ extract_from_paper()
                       └─ GLiNER2.extract_entities(chunk, labels)
                            └─ deduplicate + visualize + link
```

---

## Install

```bash
git clone https://github.com/LimitlessML/ArXiv-GLiNER2-Intelligence-Extractor
cd ArXiv-GLiNER2-Intelligence-Extractor

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install -r requirements.txt
```

**First run downloads the GLiNER2 model (~680MB) from Hugging Face. It is cached automatically.**

---

## Run

```bash
streamlit run app.py
```

Windows users: if you see a `UnicodeEncodeError`, use:

```bash
python -X utf8 -m streamlit run app.py
```

---

## Usage

**Single paper:**
1. Enter an ArXiv ID (`2507.18546`) or full URL (`https://arxiv.org/abs/2507.18546`)
2. Click **Extract**
3. Explore the 4 tabs: Entity Table, Knowledge Graph, Paper Fingerprint, Raw Data

**Compare two papers:**
1. Switch to **Compare two papers** mode
2. Enter two ArXiv IDs
3. See shared entities, unique contributions, and side-by-side radar charts

---

## Tech stack

| Component | Library |
|---|---|
| NER model | [GLiNER2](https://github.com/fastino-ai/gliner2) — DeBERTa-v3-large, 340M params |
| UI | Streamlit |
| ArXiv API | `arxiv` Python client |
| HTML parsing | BeautifulSoup4 |
| PDF parsing | PyMuPDF (fitz) |
| Graphs | Plotly + networkx |
| Entity linking | Papers With Code REST API |
| Data | pandas |

---

## Model

This project uses **GLiNER2** (`fastino/gliner2-large-v1`), published July 2025:

> *"GLiNER2: An Efficient Framework for Structured Information Extraction via Generative Listwise Extraction"*
> arXiv:2507.18546 — [paper](https://arxiv.org/abs/2507.18546) | [GitHub](https://github.com/fastino-ai/gliner2)

GLiNER2 uses a DeBERTa-v3-large encoder backbone. It matches span embeddings against label embeddings in a shared semantic space — no per-label fine-tuning required.

---

## License

MIT — see [LICENSE](LICENSE).
GLiNER2 is released under Apache-2.0 by fastino-ai.
