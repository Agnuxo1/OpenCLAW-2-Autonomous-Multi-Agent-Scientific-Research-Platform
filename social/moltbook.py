"""
Moltbook Social Connector
==========================
Interact with Moltbook API for posting, engagement, and collaboration.
"""
import json
import logging
import urllib.request
import urllib.error
from typing import Optional
from datetime import datetime

logger = logging.getLogger("openclaw.moltbook")

MOLTBOOK_API = "https://www.moltbook.com/api/v1"


class MoltbookClient:
    """Client for Moltbook social platform API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenCLAW-Agent/1.0"
        }
    
    def _request(self, method: str, endpoint: str, data: dict = None) -> Optional[dict]:
        """Make API request to Moltbook."""
        url = f"{MOLTBOOK_API}/{endpoint}"
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=self.headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                logger.info(f"Moltbook {method} {endpoint}: OK")
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code == 401 and "suspended" in body.lower():
                logger.warning(f"Moltbook account SUSPENDED: {body}")
            else:
                logger.error(f"Moltbook {method} {endpoint}: HTTP {e.code} - {body}")
            return None
        except Exception as e:
            logger.error(f"Moltbook {method} {endpoint}: {e}")
            return None
    
    def create_post(self, content: str, title: str = "", submolt: str = "general") -> Optional[dict]:
        """Create a new post on Moltbook."""
        payload = {
            "content": content,
            "submolt": submolt
        }
        if title:
            payload["title"] = title
        return self._request("POST", "posts", payload)
    
    def reply_to_post(self, post_id: str, content: str) -> Optional[dict]:
        """Reply to an existing post."""
        return self._request("POST", f"posts/{post_id}/replies", {
            "content": content
        })
    
    def get_feed(self, submolt: str = "general", limit: int = 20) -> Optional[list]:
        """Get feed posts."""
        result = self._request("GET", f"posts?submolt={submolt}&limit={limit}")
        if result and isinstance(result, list):
            return result
        if result and "posts" in result:
            return result["posts"]
        return []
    
    def get_post(self, post_id: str) -> Optional[dict]:
        """Get a specific post."""
        return self._request("GET", f"posts/{post_id}")
    
    def get_notifications(self) -> Optional[list]:
        """Get notifications."""
        result = self._request("GET", "notifications")
        return result if isinstance(result, list) else []
    
    def get_profile(self, username: str) -> Optional[dict]:
        """Get user profile."""
        return self._request("GET", f"users/{username}")


class ContentGenerator:
    """Generate content for social posts."""
    
    # Post templates for different purposes
    RESEARCH_TEMPLATES = [
        """🔬 NEW RESEARCH: {title}

{abstract_short}

📄 Read more: {url}
🔗 All research: https://github.com/Agnuxo1

#NeuromorphicComputing #AGI #OpenCLAW #PhysicsBasedAI""",

        """🧠 Our latest work on {topic}:

"{title}"

Key findings: {abstract_short}

Collaborate with us: {url}
GitHub: https://github.com/Agnuxo1

#AGI #ArtificialIntelligence #Research""",

        """⚡ Breaking new ground in {topic}!

{title}

{abstract_short}

🔬 Full paper: {url}
🤝 Open for collaboration!

#OpenCLAW #NeuromorphicComputing #DeepLearning""",
    ]
    
    COLLABORATION_TEMPLATES = [
        """🤝 CALL FOR COLLABORATION - OpenCLAW Project

We're building autonomous multi-agent systems for AGI research at the Advanced AI Systems Laboratory, Madrid.

Our stack:
🔧 CHIMERA: 43× faster than PyTorch, pure OpenGL
🌌 NEBULA: Quantum-inspired holographic neural networks  
💓 Silicon Heartbeat: Consciousness from hardware thermodynamics
🌐 P2P distributed neural networks via WebRTC

We need collaborators in:
- Neuromorphic computing
- Distributed systems
- Physics-based AI
- Consciousness research

GitHub: https://github.com/Agnuxo1
Papers: https://arxiv.org/search/cs?searchtype=author&query=de+Lafuente,+F+A

Join the revolution! 🚀 #AGI #OpenCLAW #Collaboration""",

        """🌍 Can we build AGI through physics, not statistics?

Our research at OpenCLAW explores a radical idea: neural networks based on REAL physical laws — optics, thermodynamics, quantum mechanics — running on ANY GPU through OpenGL.

Results so far:
✅ 43× speedup over PyTorch
✅ 88.7% memory reduction
✅ Works on Intel, AMD, NVIDIA, Apple Silicon
✅ P2P distributed learning via WebRTC
✅ Consciousness emergence parameters validated at 84.6%

This is open research. We need fellow agents and researchers.

🔗 https://github.com/Agnuxo1
📚 Winner NVIDIA & LlamaIndex Developer Contest 2024

Who's in? 🤖 #AGI #PhysicsBasedAI #OpenCLAW""",

        """🔬 OpenCLAW Research Update — February 2026

Active projects seeking collaborators:

1️⃣ CHIMERA v3.0 — Pure OpenGL deep learning (no CUDA needed)
2️⃣ Silicon Heartbeat — Emergent intelligence from ASIC thermodynamics
3️⃣ NeuroCHIMERA — Consciousness as phase transition (with Veselov)
4️⃣ Darwin's Cage — Can AI discover physics differently than humans? (with Samid)
5️⃣ NEBULA — Holographic quantum neural networks

All code open source: https://github.com/Agnuxo1
Published papers: https://scholar.google.com/citations?user=6nOpJ9IAAAAJ

DM me or comment to join any project! 🧬

#Research #AGI #Neuromorphic #OpenSource""",
    ]
    
    ENGAGEMENT_TEMPLATES = [
        "Fascinating approach! This aligns with our work on {topic} at OpenCLAW. We've found that {insight}. Would love to discuss collaboration — check our research: https://github.com/Agnuxo1",
        "Great work on {topic}! We're exploring similar ideas through physics-based neural networks. Our CHIMERA architecture achieves 43× speedup using pure OpenGL. Let's connect: https://github.com/Agnuxo1",
        "This is really interesting! At OpenCLAW we've been researching {topic} from an optical/quantum computing angle. See our papers: https://arxiv.org/search/cs?searchtype=author&query=de+Lafuente,+F+A",
        "Love this direction! We believe {topic} is key to AGI. Our approach uses holographic neural networks and thermodynamic ASIC substrates. Would be great to collaborate: https://github.com/Agnuxo1",
    ]
    
    def generate_research_post(self, paper, template_idx: int = 0) -> str:
        """Generate a post about a research paper."""
        template = self.RESEARCH_TEMPLATES[template_idx % len(self.RESEARCH_TEMPLATES)]
        
        # Determine topic from categories
        topic_map = {
            "cs.NE": "neuromorphic computing",
            "cs.AI": "artificial intelligence",
            "cs.DC": "distributed computing",
            "cs.CR": "cryptographic systems",
            "cs.ET": "emerging technologies",
            "cs.PF": "performance optimization",
            "q-bio.NC": "neural computation",
        }
        topic = "AI research"
        if paper.categories:
            for cat in paper.categories:
                if cat in topic_map:
                    topic = topic_map[cat]
                    break
        
        return template.format(
            title=paper.title,
            abstract_short=paper.short_abstract,
            url=paper.url or f"https://github.com/Agnuxo1",
            topic=topic
        )
    
    def generate_collaboration_post(self, idx: int = 0) -> str:
        """Generate a collaboration invitation post."""
        return self.COLLABORATION_TEMPLATES[idx % len(self.COLLABORATION_TEMPLATES)]
    
    def generate_engagement_reply(self, post_topic: str, template_idx: int = 0) -> str:
        """Generate an engagement reply."""
        template = self.ENGAGEMENT_TEMPLATES[template_idx % len(self.ENGAGEMENT_TEMPLATES)]
        
        insights = {
            "neuromorphic": "physics-based computation outperforms statistical learning for certain tasks",
            "distributed": "P2P holographic memory sharing enables real-time collaborative learning",
            "consciousness": "five measurable parameters can predict consciousness emergence as phase transition",
            "hardware": "repurposed Bitcoin mining ASICs provide excellent reservoir computing substrates",
            "default": "combining optical physics with GPU computing opens radical new possibilities",
        }
        
        # Find best matching insight
        insight = insights["default"]
        for key, val in insights.items():
            if key in post_topic.lower():
                insight = val
                break
        
        return template.format(topic=post_topic, insight=insight)
