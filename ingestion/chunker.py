import json
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
        tag = el.name
        text = el.get_text(strip=True)
        if not text:
            continue
        if tag in ("h1",):
            lines.append(f"# {text}")
        elif tag in ("h2",):
            lines.append(f"## {text}")
        elif tag in ("h3",):
            lines.append(f"### {text}")
        elif tag in ("h4", "h5", "h6"):
            prefix = "#" * int(tag[1])
            lines.append(f"{prefix} {text}")
        elif tag == "li":
            lines.append(f"- {text}")
        elif tag == "pre":
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text)

    return "\n\n".join(lines)


def chunk_pages(
    pages: list[dict],
    framework: str,
    version: str = "latest",
) -> list[dict]:
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []

    for page in pages:
        markdown = html_to_markdown(page["html"])
        header_chunks = markdown_splitter.split_text(markdown)

        for header_chunk in header_chunks:
            sub_chunks = text_splitter.split_text(header_chunk.page_content)

            for sub_chunk in sub_chunks:
                chunk_id = hashlib.sha256(
                    f"{page['url']}:{sub_chunk[:100]}".encode()
                ).hexdigest()[:32]

                metadata = {
                    **header_chunk.metadata,
                    "framework": framework,
                    "version": version,
                    "url": page["url"],
                    "title": page.get("title", ""),
                }

                chunks.append({
                    "id": chunk_id,
                    "content": sub_chunk,
                    "metadata": metadata,
                })

    return chunks


def process_framework(
    framework: str,
    input_dir: str = "data/raw",
    output_dir: str = "data/chunks",
    version: str = "latest",
):
    input_path = Path(input_dir) / framework / "pages.json"
    output_path = Path(output_dir) / framework
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages, framework, version)

    output_file = output_path / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(chunks)} chunks from {framework} → {output_file}")
    return chunks


if __name__ == "__main__":
    import sys
    framework = sys.argv[1] if len(sys.argv) > 1 else "langchain"
    process_framework(framework)
