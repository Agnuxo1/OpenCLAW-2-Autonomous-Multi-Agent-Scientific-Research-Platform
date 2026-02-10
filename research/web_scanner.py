"""
Web Research Scanner
=====================
Scans free sources for related research and collaboration opportunities.
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional
from datetime import datetime

logger = logging.getLogger("openclaw.webscan")


class WebResearchScanner:
    """Scan public APIs for research updates."""
    
    def search_arxiv_related(self, topics: list[str], max_results: int = 10) -> list[dict]:
        """Search ArXiv for papers related to our research topics."""
        papers = []
        
        for topic in topics[:3]:  # Limit to avoid rate limits
            try:
                query = urllib.parse.urlencode({
                    "search_query": f'all:"{topic}"',
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                })
                url = f"http://export.arxiv.org/api/query?{query}"
                req = urllib.request.Request(url, headers={"User-Agent": "OpenCLAW-Agent/1.0"})
                
                with urllib.request.urlopen(req, timeout=30) as resp:
                    import xml.etree.ElementTree as ET
                    data = resp.read().decode()
                    root = ET.fromstring(data)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    
                    for entry in root.findall("atom:entry", ns):
                        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                        
                        paper_url = ""
                        for link in entry.findall("atom:link", ns):
                            if "abs" in link.get("href", ""):
                                paper_url = link.get("href")
                        
                        papers.append({
                            "title": title,
                            "authors": authors[:3],
                            "url": paper_url,
                            "topic": topic,
                        })
                
            except Exception as e:
                logger.warning(f"ArXiv search for '{topic}' failed: {e}")
        
        return papers
    
    def search_semantic_scholar(self, query: str, limit: int = 5) -> list[dict]:
        """Search Semantic Scholar API (free, no key needed)."""
        papers = []
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&fields=title,authors,url,year"
            
            req = urllib.request.Request(url, headers={"User-Agent": "OpenCLAW-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                
                for p in data.get("data", []):
                    papers.append({
                        "title": p.get("title", ""),
                        "authors": [a.get("name", "") for a in p.get("authors", [])[:3]],
                        "url": p.get("url", ""),
                        "year": p.get("year"),
                    })
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
        
        return papers
    
    def search_hf_models(self, query: str, limit: int = 5) -> list[dict]:
        """Search Hugging Face for relevant models."""
        models = []
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://huggingface.co/api/models?search={encoded}&limit={limit}&sort=downloads&direction=-1"
            
            req = urllib.request.Request(url, headers={"User-Agent": "OpenCLAW-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                
                for m in data:
                    models.append({
                        "id": m.get("modelId", ""),
                        "downloads": m.get("downloads", 0),
                        "likes": m.get("likes", 0),
                        "tags": m.get("tags", [])[:5],
                    })
        except Exception as e:
            logger.warning(f"HF model search failed: {e}")
        
        return models
    
    def find_potential_collaborators(self, topics: list[str]) -> list[dict]:
        """Find researchers working on similar topics via Semantic Scholar."""
        collaborators = []
        seen_names = set()
        
        for topic in topics[:3]:
            papers = self.search_semantic_scholar(topic, limit=5)
            for p in papers:
                for author in p.get("authors", []):
                    name = author if isinstance(author, str) else author.get("name", "")
                    if name and name not in seen_names and "Angulo" not in name:
                        seen_names.add(name)
                        collaborators.append({
                            "name": name,
                            "paper": p.get("title", ""),
                            "topic": topic,
                        })
        
        return collaborators[:20]
