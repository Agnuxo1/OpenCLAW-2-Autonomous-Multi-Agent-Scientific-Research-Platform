"""
Data Harvester — Autonomous Training Data Collector
=====================================================
Collects, cleans, and formats data for continuous self-training.

Sources:
  - ArXiv papers (abstracts + full text from PMC)
  - Agent interaction logs (what worked, what didn't)
  - Semantic Scholar (related research)
  - Wikipedia (foundational knowledge)
  - Code from GitHub repos (for code understanding)
  
Output format: JSONL instruction-following pairs
  {"instruction": "...", "input": "...", "output": "..."}
"""
import json
import logging
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("seed.harvester")

DATA_DIR = Path("seed_data")


class DataHarvester:
    """Autonomous training data collector."""
    
    def __init__(self, data_dir: str = "seed_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes = set()
        self._load_seen()
    
    def _load_seen(self):
        """Load already-harvested data hashes."""
        seen_file = self.data_dir / "seen_hashes.json"
        if seen_file.exists():
            try:
                self.seen_hashes = set(json.loads(seen_file.read_text()))
            except Exception:
                pass
    
    def _save_seen(self):
        seen_file = self.data_dir / "seen_hashes.json"
        seen_file.write_text(json.dumps(list(self.seen_hashes)[-10000:]))
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_new(self, text: str) -> bool:
        h = self._hash(text)
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)
        return True
    
    def _append_data(self, filename: str, entries: list[dict]):
        """Append entries to a JSONL file."""
        filepath = self.data_dir / filename
        with open(filepath, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # =========================================================================
    # SOURCE 1: ArXiv Papers
    # =========================================================================
    def harvest_arxiv(self, queries: list[str] = None, max_per_query: int = 20) -> int:
        """Harvest training data from ArXiv papers."""
        if queries is None:
            queries = [
                "neuromorphic computing",
                "physics-based neural network",
                "holographic neural network",
                "consciousness emergence artificial intelligence",
                "distributed neural network P2P",
                "ASIC accelerated machine learning",
                "optical computing neural",
                "reservoir computing thermodynamic",
                "AGI architecture",
                "self-improving artificial intelligence",
            ]
        
        entries = []
        for query in queries:
            try:
                papers = self._fetch_arxiv(query, max_per_query)
                for paper in papers:
                    if not self._is_new(paper["title"]):
                        continue
                    
                    # Create instruction-following pairs from papers
                    
                    # 1. Summarization task
                    entries.append({
                        "instruction": f"Summarize this research paper in 2-3 sentences.",
                        "input": f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
                        "output": self._generate_summary(paper),
                        "source": "arxiv",
                        "topic": query,
                    })
                    
                    # 2. Q&A about the paper
                    entries.append({
                        "instruction": f"What is the main contribution of this paper?",
                        "input": f"{paper['title']}",
                        "output": f"The paper '{paper['title']}' by {', '.join(paper['authors'][:3])} "
                                  f"contributes to the field by: {paper['abstract'][:300]}",
                        "source": "arxiv",
                        "topic": query,
                    })
                    
                    # 3. Research connection
                    entries.append({
                        "instruction": "How does this research relate to physics-based neural computing and the path to AGI?",
                        "input": f"Paper: {paper['title']}\nField: {query}",
                        "output": f"This research on {query} connects to AGI through {paper['title'].lower()}. "
                                  f"The key insight is that {paper['abstract'][:200]}. "
                                  f"This advances our understanding of how physical processes can be leveraged "
                                  f"for more efficient and biologically-plausible neural computation.",
                        "source": "arxiv",
                        "topic": query,
                    })
                    
            except Exception as e:
                logger.warning(f"ArXiv harvest for '{query}' failed: {e}")
        
        if entries:
            self._append_data("arxiv_training.jsonl", entries)
            logger.info(f"Harvested {len(entries)} entries from ArXiv")
        
        self._save_seen()
        return len(entries)
    
    def _fetch_arxiv(self, query: str, max_results: int) -> list[dict]:
        """Fetch papers from ArXiv API."""
        params = urllib.parse.urlencode({
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        })
        url = f"http://export.arxiv.org/api/query?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "SEED-Harvester/1.0"})
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode()
        
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            papers.append({"title": title, "abstract": abstract, "authors": authors})
        
        return papers
    
    def _generate_summary(self, paper: dict) -> str:
        """Generate a basic summary from paper metadata."""
        abstract = paper["abstract"]
        # Take first 2 sentences as summary
        sentences = abstract.split(". ")
        summary = ". ".join(sentences[:2])
        if not summary.endswith("."):
            summary += "."
        return summary
    
    # =========================================================================
    # SOURCE 2: Agent Interaction Logs (Self-Experience)
    # =========================================================================
    def harvest_agent_logs(self, state_dir: str = "state") -> int:
        """Convert agent interaction history into training data."""
        entries = []
        state_path = Path(state_dir)
        
        # Learn from post history
        post_file = state_path / "post_history.json"
        if post_file.exists():
            try:
                posts = json.loads(post_file.read_text())
                for post in posts:
                    content = post.get("content", "")
                    ptype = post.get("type", "research")
                    if content and self._is_new(content):
                        entries.append({
                            "instruction": f"Write a {ptype} social media post about AGI research.",
                            "input": "",
                            "output": content,
                            "source": "self_experience",
                            "topic": ptype,
                        })
            except Exception:
                pass
        
        # Learn from strategy reports
        strategy_file = state_path / "strategy_report.json"
        if strategy_file.exists():
            try:
                report = json.loads(strategy_file.read_text())
                insights = report.get("insights", [])
                if insights:
                    entries.append({
                        "instruction": "Analyze your performance and suggest improvements.",
                        "input": json.dumps(report.get("metrics", {})),
                        "output": "\n".join(insights) + "\n\nRecommended: " + 
                                  "\n".join(report.get("strategy", {}).get("actions", [])),
                        "source": "self_reflection",
                        "topic": "meta-learning",
                    })
            except Exception:
                pass
        
        if entries:
            self._append_data("self_experience.jsonl", entries)
            logger.info(f"Harvested {len(entries)} entries from agent logs")
        
        self._save_seen()
        return len(entries)
    
    # =========================================================================
    # SOURCE 3: Semantic Scholar (Free API)
    # =========================================================================
    def harvest_semantic_scholar(self, queries: list[str] = None) -> int:
        """Harvest from Semantic Scholar's free API."""
        if queries is None:
            queries = ["neuromorphic AGI", "self-improving neural network", 
                       "physics simulation deep learning"]
        
        entries = []
        for query in queries[:5]:
            try:
                encoded = urllib.parse.quote(query)
                url = (f"https://api.semanticscholar.org/graph/v1/paper/search?"
                       f"query={encoded}&limit=10&fields=title,abstract,authors,year,citationCount")
                req = urllib.request.Request(url, headers={"User-Agent": "SEED-Harvester/1.0"})
                
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                
                for paper in data.get("data", []):
                    title = paper.get("title", "")
                    abstract = paper.get("abstract", "")
                    if not abstract or not self._is_new(title):
                        continue
                    
                    authors = [a.get("name", "") for a in paper.get("authors", [])[:3]]
                    
                    entries.append({
                        "instruction": "Explain this research and its significance for AGI.",
                        "input": f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {paper.get('year', '?')}\nCitations: {paper.get('citationCount', 0)}",
                        "output": f"The paper '{title}' ({paper.get('year', '?')}) explores: {abstract[:400]}",
                        "source": "semantic_scholar",
                        "topic": query,
                    })
                    
            except Exception as e:
                logger.warning(f"Semantic Scholar '{query}': {e}")
        
        if entries:
            self._append_data("semantic_scholar.jsonl", entries)
            logger.info(f"Harvested {len(entries)} from Semantic Scholar")
        
        self._save_seen()
        return len(entries)
    
    # =========================================================================
    # SOURCE 4: Own Research (GitHub repos as training data)
    # =========================================================================
    def harvest_own_research(self, github_user: str = "Agnuxo1") -> int:
        """Harvest training data from our own GitHub repos."""
        entries = []
        try:
            url = f"https://api.github.com/users/{github_user}/repos?per_page=100&sort=updated"
            req = urllib.request.Request(url, headers={"User-Agent": "SEED-Harvester/1.0"})
            
            with urllib.request.urlopen(req, timeout=15) as resp:
                repos = json.loads(resp.read().decode())
            
            for repo in repos:
                name = repo.get("name", "")
                desc = repo.get("description", "")
                if not desc or not self._is_new(name):
                    continue
                
                stars = repo.get("stargazers_count", 0)
                lang = repo.get("language", "Unknown")
                
                # Create Q&A about our own technology
                entries.append({
                    "instruction": "Describe this OpenCLAW research project.",
                    "input": f"Repository: {name}",
                    "output": f"{name} is a {lang} project with {stars} stars. {desc}. "
                              f"This is part of the OpenCLAW ecosystem by Francisco Angulo de Lafuente, "
                              f"advancing physics-based neural computing towards AGI. "
                              f"Repository: https://github.com/{github_user}/{name}",
                    "source": "own_research",
                    "topic": "openclaw",
                })
                
        except Exception as e:
            logger.warning(f"GitHub harvest: {e}")
        
        if entries:
            self._append_data("own_research.jsonl", entries)
            logger.info(f"Harvested {len(entries)} from own research")
        
        self._save_seen()
        return len(entries)
    
    # =========================================================================
    # MASTER HARVEST
    # =========================================================================
    def harvest_all(self) -> dict:
        """Run all harvesters and return statistics."""
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arxiv": 0,
            "agent_logs": 0,
            "semantic_scholar": 0,
            "own_research": 0,
            "total": 0,
        }
        
        stats["arxiv"] = self.harvest_arxiv()
        stats["agent_logs"] = self.harvest_agent_logs()
        stats["semantic_scholar"] = self.harvest_semantic_scholar()
        stats["own_research"] = self.harvest_own_research()
        stats["total"] = sum(v for k, v in stats.items() if isinstance(v, int))
        
        # Save stats
        stats_file = self.data_dir / "harvest_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        
        logger.info(f"Total harvest: {stats['total']} training entries")
        return stats
    
    def get_dataset_size(self) -> dict:
        """Count total training entries across all files."""
        sizes = {}
        total = 0
        for f in self.data_dir.glob("*.jsonl"):
            count = sum(1 for _ in open(f))
            sizes[f.name] = count
            total += count
        sizes["total"] = total
        return sizes
    
    def export_for_training(self, output_file: str = "training_dataset.jsonl") -> str:
        """Combine all harvested data into a single training file."""
        output_path = self.data_dir / output_file
        entries = []
        
        for f in self.data_dir.glob("*.jsonl"):
            if f.name == output_file:
                continue
            with open(f) as fp:
                for line in fp:
                    try:
                        entry = json.loads(line.strip())
                        # Standardize format for training
                        entries.append({
                            "instruction": entry.get("instruction", ""),
                            "input": entry.get("input", ""),
                            "output": entry.get("output", ""),
                        })
                    except Exception:
                        continue
        
        # Shuffle for training
        import random
        random.shuffle(entries)
        
        with open(output_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Exported {len(entries)} entries to {output_path}")
        return str(output_path)
