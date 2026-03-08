import requests

LINKABLE_LABELS = {"ML model", "dataset", "benchmark", "method or technique"}

PWC_API = "https://paperswithcode.com/api/v1"
PWC_LABEL_MAP = {
    "ML model":            "methods",
    "method or technique": "methods",
    "dataset":             "datasets",
    "benchmark":           "datasets",
}


def _search_pwc(entity_text: str, pwc_type: str) -> str | None:
    try:
        resp = requests.get(
            f"{PWC_API}/{pwc_type}/",
            params={"q": entity_text, "limit": 1},
            timeout=4,
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        slug = results[0].get("id") or results[0].get("slug")
        if slug:
            kind = "method" if pwc_type == "methods" else "dataset"
            return f"https://paperswithcode.com/{kind}/{slug}"
    except Exception:
        return None
    return None


def link_entities(entities: list[dict]) -> list[dict]:
    linked = []
    for e in entities:
        url = None
        if e["label"] in LINKABLE_LABELS:
            pwc_type = PWC_LABEL_MAP[e["label"]]
            url = _search_pwc(e["text"], pwc_type)
        linked.append({**e, "pwc_url": url})
    return linked
