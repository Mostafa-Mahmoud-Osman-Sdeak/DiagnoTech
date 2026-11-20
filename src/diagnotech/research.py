"""Trusted external medical research (spec section 9).

Used only when the local knowledge base cannot answer. Searches are
restricted to approved medical domains (WHO, NHS, CDC, Mayo Clinic) via a
site-filtered web search, result pages are fetched and reduced to text,
and every item keeps its source URL so the answer can cite it and clearly
mark it as external. Any failure returns an empty list; the caller then
falls back to the "unavailable" answer state. The application works fully
offline with this component disabled or unreachable.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

logger = logging.getLogger(__name__)

APPROVED_DOMAINS = ("who.int", "nhs.uk", "cdc.gov", "mayoclinic.org")
_SEARCH_URL = "https://html.duckduckgo.com/html/?q="
_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)


@dataclass
class ExternalEvidence:
    title: str
    url: str
    text: str
    domain: str

    def evidence_block(self) -> str:
        return f"المصدر الخارجي: {self.domain} ({self.url})\nالعنوان: {self.title}\n{self.text}"

    def source_ref(self) -> dict:
        return {"record_id": "", "source": f"مصدر خارجي: {self.domain}",
                "source_url": self.url, "disease_name": self.title,
                "category": "External", "score": 0.0}


def _strip_html(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _unwrap_ddg_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc:
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _extract_page_text(page_html: str, limit: int = 2500) -> str:
    body = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", page_html)
    text = _strip_html(body)
    return text[:limit]


class WebResearcher:
    def __init__(self, timeout: float = 10.0, max_results: int = 3):
        self.timeout = timeout
        self.max_results = max_results
        self._cache: dict[str, list[ExternalEvidence]] = {}

    def search(self, question_en: str) -> list[ExternalEvidence]:
        """Search approved domains. ``question_en`` should be an English
        rendering of the user's question for reliable results."""
        if question_en in self._cache:
            return self._cache[question_en]
        results = self._search_once(question_en)
        self._cache[question_en] = results
        return results

    def _search_once(self, question_en: str) -> list[ExternalEvidence]:
        import httpx

        site_filter = " OR ".join(f"site:{d}" for d in APPROVED_DOMAINS)
        query = quote_plus(f"{question_en} ({site_filter})")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DiagnoTech/1.0)"}
        try:
            with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(_SEARCH_URL + query)
                response.raise_for_status()
                links = _LINK_RE.findall(response.text)
                snippets = _SNIPPET_RE.findall(response.text)
                evidence: list[ExternalEvidence] = []
                for i, (raw_url, raw_title) in enumerate(links):
                    url = _unwrap_ddg_url(raw_url)
                    domain = urlparse(url).netloc.lower()
                    if not any(domain.endswith(d) for d in APPROVED_DOMAINS):
                        continue
                    title = _strip_html(raw_title)
                    snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
                    page_text = ""
                    try:
                        page = client.get(url)
                        if page.status_code == 200:
                            page_text = _extract_page_text(page.text)
                    except Exception:
                        page_text = ""
                    text = page_text or snippet
                    if text:
                        evidence.append(ExternalEvidence(
                            title=title, url=url, text=text, domain=domain))
                    if len(evidence) >= self.max_results:
                        break
                return evidence
        except Exception as exc:
            logger.warning("external research failed: %s", exc)
            return []
