

import requests
import xml.etree.ElementTree as ET

from config import (
    ENABLE_WEB_SEARCH,
    PUBMED_MAX_RESULTS,
    MEDLINEPLUS_MAX_RESULTS,
    TAVILY_API_KEY,
    WEB_REQUEST_TIMEOUT,
)


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS):
    """Returns list of {"source", "text", "url"} from PubMed abstracts."""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        r = requests.get(search_url, params=params, timeout=WEB_REQUEST_TIMEOUT)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        r = requests.get(fetch_url, params=params, timeout=WEB_REQUEST_TIMEOUT)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        results = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            pmid_el = article.find(".//PMID")
            title = title_el.text if title_el is not None else "Untitled"
            abstract = abstract_el.text if abstract_el is not None else ""
            pmid = pmid_el.text if pmid_el is not None else ""
            if not abstract:
                continue
            results.append({
                "source": f"PubMed: {title}",
                "text": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return results
    except Exception as e:
        print(f"[web_search] PubMed lookup failed: {e}")
        return []



def search_medlineplus(query: str, max_results: int = MEDLINEPLUS_MAX_RESULTS):
    """Returns list of {"source", "text", "url"} from MedlinePlus health topics."""
    try:
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": query, "retmax": max_results}
        r = requests.get(url, params=params, timeout=WEB_REQUEST_TIMEOUT)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        results = []
        for doc in root.findall(".//document"):
            url_attr = doc.get("url", "")
            title, snippet = "", ""
            for content in doc.findall("content"):
                name = content.get("name")
                text = "".join(content.itertext()).strip()
                if name == "title":
                    title = text
                elif name == "FullSummary":
                    snippet = text
            if not snippet:
                continue
            results.append({
                "source": f"MedlinePlus: {title}",
                "text": snippet,
                "url": url_attr,
            })
        return results
    except Exception as e:
        print(f"[web_search] MedlinePlus lookup failed: {e}")
        return []


def search_web_generic(query: str, max_results: int = 3):
    if not TAVILY_API_KEY:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=WEB_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "source": f"Web: {item.get('title', 'result')}",
                "text": item.get("content", ""),
                "url": item.get("url", ""),
            })
        return results
    except Exception as e:
        print(f"[web_search] generic web search failed: {e}")
        return []


def fetch_web_context(query: str):
    """Aggregates results from all enabled sources. Fails soft per-source."""
    if not ENABLE_WEB_SEARCH:
        return []
    results = []
    results.extend(search_pubmed(query))
    results.extend(search_medlineplus(query))
    results.extend(search_web_generic(query))
    return results
