"""Download the Indian labour codes and split them into retrievable passages."""
import os, re, json, requests, urllib3
import fitz

urllib3.disable_warnings()

SOURCES = {
    "osh": "https://dgfasli.gov.in/public/Admin/Cms/AllPdf/OSH_Gazette.pdf",
    "industrial_relations": "https://prsindia.org/files/bills_acts/acts_parliament/2020/Industrial%20Relations%20Code,%202020.pdf",
    "social_security": "https://www.labour.gov.in/static/uploads/2025/07/b0620548445580767b5c0d18c95c26f7.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Gazette boilerplate repeats on every page. Left in, it becomes the most common
# phrase in the corpus and blurs the distinction between passages.
BOILERPLATE = [
    r"THE GAZETTE OF INDIA EXTRAORDINARY",
    r"\[PART II\s*[\u2014-]?",
    r"SEC\.\s*1\]",
    r"PUBLISHED BY AUTHORITY",
    r"REGISTERED NO\..*",
    r"xxxGID[EH]xxx",
    r"[\u0900-\u097F]+",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def strip_boilerplate(text):
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed-size windows, preferring to break at a sentence end.

    Overlap matters: a provision sitting on a boundary would otherwise be cut in
    half and lost from both passages. Repeating the tail gives it two chances.
    """
    passages, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            breaks = list(re.finditer(r"[.;:]\s", text[end - 200:end]))
            if breaks:
                end = end - 200 + breaks[-1].end()
        passage = text[start:end].strip()
        if len(passage) > 100:
            passages.append(passage)
        if end >= len(text):
            break
        start = end - overlap
    return passages


def download(name, url, folder="docs"):
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{name}.pdf"
    if os.path.exists(path):
        return path
    response = requests.get(url, headers=HEADERS, timeout=60, verify=False)
    if response.content[:4] != b"%PDF":
        raise ValueError(f"{name}: not a PDF")
    open(path, "wb").write(response.content)
    return path


def build(out="data/chunks.jsonl"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    chunks = []
    for name, url in SOURCES.items():
        pdf = fitz.open(download(name, url))
        for page_no in range(len(pdf)):
            text = strip_boilerplate(pdf[page_no].get_text())
            for i, passage in enumerate(split(text)):
                chunks.append({
                    "id": f"{name}_p{page_no + 1}_c{i}",
                    "text": passage,
                    "code": name,
                    "page": page_no + 1,
                })
        pdf.close()
    with open(out, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return chunks


if __name__ == "__main__":
    print(f"{len(build()):,} chunks written")
