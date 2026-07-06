from collections import deque
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import json
import time
from pathlib import Path


DOC_SOURCES = {
    "langchain": {
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "allowed_prefixes": ["https://docs.langchain.com/oss/python/"],
        "exclude_prefixes": ["https://docs.langchain.com/oss/python/reference/"],
    },
    "llamaindex": {
        "url": "https://developers.llamaindex.ai/python/framework/",
        "allowed_prefixes": ["https://developers.llamaindex.ai/python/framework/"],
        "exclude_prefixes": ["https://developers.llamaindex.ai/python/framework/api/"],
    },
    "haystack": {
        "url": "https://docs.haystack.deepset.ai/docs",
        "allowed_prefixes": ["https://docs.haystack.deepset.ai/"],
        "exclude_prefixes": ["https://docs.haystack.deepset.ai/api/"],
    },
}


def scrape_docs(framework: str, output_dir: str = "data/raw"):
    config = DOC_SOURCES.get(framework)
    if not config:
        raise ValueError(f"Unknown framework: {framework}")

    output_path = Path(output_dir) / framework
    output_path.mkdir(parents=True, exist_ok=True)

    visited = set()
    queued = set()
    to_visit = deque([config["url"]])
    queued.add(config["url"])
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        max_pages = 50

        while to_visit and len(visited) < max_pages:
            url = to_visit.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to load {url}: {e}")
                continue

            title = page.title()
            content = page.content()
            print(f"  [{len(visited)}] {title[:60]}", flush=True)

            pages.append({
                "url": url,
                "title": title,
                "html": content,
            })

            links = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(el => el.href)",
            )

            for link in links:
                normalized = urljoin(url, link)
                parsed = urlparse(normalized)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"

                if not any(clean_url.startswith(p) for p in config["allowed_prefixes"]):
                    continue
                if any(clean_url.startswith(e) for e in config["exclude_prefixes"]):
                    continue
                if clean_url not in visited and clean_url not in queued:
                    queued.add(clean_url)
                    to_visit.append(clean_url)

        browser.close()

    output_file = output_path / "pages.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Scraped {len(pages)} pages from {framework} → {output_file}")
    return pages


if __name__ == "__main__":
    import sys
    framework = sys.argv[1] if len(sys.argv) > 1 else "langchain"
    scrape_docs(framework)
