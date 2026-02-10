"""
Research Paper Fetcher
======================
Fetches REAL papers from ArXiv and Google Scholar.
"""
import re
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

logger = logging.getLogger("openclaw.research")


@dataclass
class Paper:
    """A research paper."""
    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str = ""
    url: str = ""
    published: str = ""
    categories: list[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
    
    @property
    def short_abstract(self) -> str:
        """First 280 chars of abstract."""
        if len(self.abstract) <= 280:
            return self.abstract
        return self.abstract[:277] + "..."
    
    @property
    def uid(self) -> str:
        return hashlib.md5(self.title.encode()).hexdigest()[:12]


class ArxivFetcher:
    """Fetch papers from ArXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # Known papers by Francisco Angulo de Lafuente
    KNOWN_PAPERS = [
        Paper(
            title="Speaking to Silicon: Neural Communication with Bitcoin Mining ASICs via Thermodynamic Probability Filtering",
            authors=["Francisco Angulo de Lafuente"],
            abstract="This paper presents a novel approach to neural communication with Bitcoin mining ASICs through thermodynamic probability filtering, enabling the extraction of meaningful patterns from hardware thermal noise for reservoir computing applications.",
            arxiv_id="2601.12032",
            url="https://arxiv.org/abs/2601.12032",
            published="2025-01",
            categories=["cs.NE", "cs.AI"]
        ),
        Paper(
            title="SiliconHealth: Blockchain-Integrated ASIC-RAG Architecture for Healthcare Data Sovereignty",
            authors=["Francisco Angulo de Lafuente", "Seid Mehammed Abdu"],
            abstract="A novel blockchain-integrated architecture combining ASIC hardware acceleration with Retrieval-Augmented Generation for healthcare data sovereignty and medical anomaly detection.",
            arxiv_id="2601.09557",
            url="https://arxiv.org/abs/2601.09557",
            published="2025-01",
            categories=["cs.CR", "cs.AI"]
        ),
        Paper(
            title="Holographic Reservoir Computing with Thermodynamic ASIC Substrates: Silicon Heartbeat for Emergent Neuromorphic Intelligence",
            authors=["Francisco Angulo de Lafuente"],
            abstract="We present a framework for emergent neuromorphic intelligence using holographic reservoir computing in thermodynamic ASIC substrates, demonstrating that repurposed Bitcoin mining hardware can serve as a substrate for emergent neural computation.",
            arxiv_id="2601.01916",
            url="https://arxiv.org/abs/2601.01916",
            published="2025-01",
            categories=["cs.NE", "cs.ET"]
        ),
        Paper(
            title="CHIMERA: Cognitive Hybrid Intelligence for Memory-Embedded Reasoning Architecture",
            authors=["Francisco Angulo de Lafuente"],
            abstract="A revolutionary neuromorphic computing system achieving 43x speedup over PyTorch with 88.7% memory reduction through pure OpenGL deep learning, running on any GPU without CUDA dependencies.",
            arxiv_id="",
            url="https://github.com/Agnuxo1/CHIMERA-Revolutionary-AI-Architecture---Pure-OpenGL-Deep-Learning",
            published="2024-12",
            categories=["cs.NE", "cs.AI", "cs.PF"]
        ),
        Paper(
            title="NeuroCHIMERA: Consciousness Emergence as Phase Transition in GPU-Native Neuromorphic Computing",
            authors=["Vladimir F. Veselov", "Francisco Angulo de Lafuente"],
            abstract="Consciousness understood as emergent phase transition when five critical parameters simultaneously exceed thresholds. 84.6% neuroscience validation accuracy. 15.7 billion HNS operations/sec on RTX 3090.",
            arxiv_id="",
            url="https://github.com/Agnuxo1/NeuroCHIMERA__GPU-Native_Neuromorphic_Consciousness",
            published="2025-12",
            categories=["cs.NE", "q-bio.NC"]
        ),
        Paper(
            title="Empirical Evidence for AI Breaking the Barrier via Optical Chaos - Darwin's Cage Experiments",
            authors=["Francisco Angulo de Lafuente", "Gideon Samid"],
            abstract="20 experimental investigations testing whether AI can discover physical laws through representations fundamentally different from human mathematical frameworks. The Darwin's Cage hypothesis.",
            arxiv_id="",
            url="https://github.com/Agnuxo1/Empirical-Evidence-for-AI-AIM-Breaking-the-Barrier-via-Optical-Chaos",
            published="2025-12",
            categories=["cs.AI", "physics.comp-ph"]
        ),
        Paper(
            title="NEBULA: Neural Entanglement-Based Unified Learning Architecture",
            authors=["Francisco Angulo de Lafuente"],
            abstract="A dynamic AI system integrating quantum computing principles and biological neural networks. Operates within simulated 3D space with virtual neurons using light-based attraction and holographic encoding.",
            arxiv_id="",
            url="https://github.com/Agnuxo1/NEBULA",
            published="2024-08",
            categories=["cs.NE", "cs.AI"]
        ),
        Paper(
            title="Enhanced Unified Holographic Neural Network (EUHNN) with P2P Distributed Learning",
            authors=["Francisco Angulo de Lafuente"],
            abstract="Winner NVIDIA & LlamaIndex Developer Contest 2024. Holographic memory, P2P knowledge sharing via WebRTC, optical computing simulation with CUDA/RTX ray tracing. Real-time distributed learning.",
            arxiv_id="",
            url="https://github.com/Agnuxo1/Unified-Holographic-Neural-Network",
            published="2024-07",
            categories=["cs.NE", "cs.DC"]
        ),
    ]
    
    def fetch_from_arxiv(self, author: str = "Angulo de Lafuente") -> list[Paper]:
        """Fetch papers from ArXiv API."""
        papers = []
        try:
            query = urllib.parse.urlencode({
                "search_query": f'au:"{author}"',
                "start": 0,
                "max_results": 20,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            })
            url = f"{self.BASE_URL}?{query}"
            
            req = urllib.request.Request(url, headers={"User-Agent": "OpenCLAW-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode()
            
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                
                arxiv_id = ""
                paper_url = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if "abs" in href:
                        paper_url = href
                        arxiv_id = href.split("/abs/")[-1]
                
                published = entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else ""
                
                categories = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    categories.append(cat.get("term", ""))
                
                papers.append(Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    url=paper_url,
                    published=published,
                    categories=categories
                ))
            
            logger.info(f"Fetched {len(papers)} papers from ArXiv")
        except Exception as e:
            logger.warning(f"ArXiv fetch failed: {e}, using known papers")
        
        # Merge with known papers (avoid duplicates)
        known_titles = {p.title.lower() for p in papers}
        for kp in self.KNOWN_PAPERS:
            if kp.title.lower() not in known_titles:
                papers.append(kp)
        
        return papers
    
    def get_all_papers(self) -> list[Paper]:
        """Get all papers (ArXiv + known)."""
        papers = self.fetch_from_arxiv()
        if not papers:
            papers = self.KNOWN_PAPERS.copy()
        return papers
