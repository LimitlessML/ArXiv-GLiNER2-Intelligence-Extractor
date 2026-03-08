from gliner2 import GLiNER2

ENTITY_LABELS = [
    "ML model",
    "dataset",
    "metric",
    "institution",
    "method or technique",
    "benchmark",
    "author",
]

def load_model(model_name: str = "fastino/gliner2-large-v1") -> GLiNER2:
    return GLiNER2.from_pretrained(model_name)

def extract_from_chunks(model: GLiNER2, chunks: list[str]) -> list[dict]:
    seen = set()
    entities = []
    for chunk in chunks:
        result = model.extract_entities(chunk, ENTITY_LABELS)
        for label, texts in result["entities"].items():
            for text in texts:
                key = (text.lower(), label)
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "text": text,
                        "label": label,
                    })
    return entities

KEY_SECTIONS = ["introduction", "method", "result", "conclusion", "experiment", "abstract"]

def _is_key_section(name: str) -> bool:
    return any(k in name.lower() for k in KEY_SECTIONS)

def extract_from_paper(model: GLiNER2, paper: dict, full_paper: bool = False) -> dict:
    results = {}
    for section_name, chunks in paper["sections"].items():
        if not full_paper and not _is_key_section(section_name):
            continue
        entities = extract_from_chunks(model, chunks)
        if entities:
            results[section_name] = entities
    return results