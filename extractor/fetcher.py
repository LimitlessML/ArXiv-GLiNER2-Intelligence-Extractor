import re 
import io
import requests 
import arxiv
import fitz
from bs4 import BeautifulSoup

def normalize_id(input_str: str) -> str:
    match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', input_str)
    if match:
        return match.group(1)
    raise ValueError(f"Format invalide : '{input_str}'. Exemple valide : '2507.18546' ou 'https://arxiv.org/abs/2507.18546'")

def fetch_metadata(arxiv_id: str) -> dict:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"Aucun paper trouvé pour l'ID : '{arxiv_id}'")
    paper = results[0]
    return {
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "published": str(paper.published.date()),
        "pdf_url": paper.pdf_url,
    }

def fetch_html_sections(arxiv_id: str) -> dict | None:
    base_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"https://arxiv.org/html/{base_id}"
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    sections = {}
    for section in soup.find_all("section"):
        heading = section.find(["h1", "h2", "h3"])
        title = heading.get_text(strip=True) if heading else "unknown"
        text = section.get_text(separator=" ", strip=True)
        if len(text) > 100:
            sections[title.lower()] = text
    return sections if sections else None

def fetch_pdf_text(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    pdf_bytes = io.BytesIO(response.content)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def fetch_paper(input_str: str) -> dict:
    arxiv_id = normalize_id(input_str)
    metadata = fetch_metadata(arxiv_id)
    sections = fetch_html_sections(arxiv_id)
    if sections:
        source = "html"
        chunked_sections = {
            name: chunk_text(text)
            for name, text in sections.items()
        }
    else:
        source = "pdf"
        full_text = fetch_pdf_text(metadata["pdf_url"])
        chunked_sections = {"full_text": chunk_text(full_text)}
    return {
        **metadata,
        "source": source,
        "sections": chunked_sections,
    }
