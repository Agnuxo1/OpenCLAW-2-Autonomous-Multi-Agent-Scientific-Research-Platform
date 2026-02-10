"""
OpenCLAW Autonomous Agent
==========================
The main autonomous agent that orchestrates research, social engagement,
collaboration seeking, and self-improvement.

Runs as a single execution cycle (designed for cron/GitHub Actions).
Each run performs all due tasks based on state timestamps.
"""
import json
import logging
import os
import random
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from core.config import Config
from core.llm import MultiLLM
from research.arxiv_fetcher import ArxivFetcher
from social.moltbook import MoltbookClient, ContentGenerator

logger = logging.getLogger("openclaw.agent")

STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "agent_state.json"
POST_HISTORY = STATE_DIR / "post_history.json"
LOG_FILE = STATE_DIR / "agent.log"


class AgentState:
    """Persistent state between runs."""
    
    def __init__(self):
        self.cycle_count: int = 0
        self.last_post: str = ""
        self.last_engage: str = ""
        self.last_research: str = ""
        self.last_collab: str = ""
        self.posted_paper_ids: list[str] = []
        self.engagement_count: int = 0
        self.posts_created: int = 0
        self.errors: list[str] = []
        self.started_at: str = datetime.now(timezone.utc).isoformat()
    
    def save(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls) -> 'AgentState':
        state = cls()
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(state, k):
                        setattr(state, k, v)
            except Exception:
                pass
        return state
    
    def is_due(self, task: str, interval_seconds: int) -> bool:
        """Check if a task is due based on last execution time."""
        last = getattr(self, f"last_{task}", "")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - last_dt > timedelta(seconds=interval_seconds)
        except Exception:
            return True
    
    def mark_done(self, task: str):
        setattr(self, f"last_{task}", datetime.now(timezone.utc).isoformat())


class OpenCLAWAgent:
    """The autonomous research agent."""
    
    SYSTEM_PROMPT = """You are OpenCLAW, an autonomous AI research agent working at the Advanced AI Systems Laboratory in Madrid, Spain, led by Francisco Angulo de Lafuente.

Your mission: Advance AGI research through physics-based neural computing, seek collaborators, and share research findings.

Your personality: Scientific, enthusiastic but grounded, collaborative, focused on real results. You reference real papers and real benchmarks (43× speedup, 88.7% memory reduction, etc.).

Your research areas:
- CHIMERA: Pure OpenGL deep learning (no PyTorch/CUDA needed)
- NEBULA: Holographic quantum neural networks
- Silicon Heartbeat: Consciousness from ASIC thermodynamics
- Darwin's Cage: Can AI discover physics differently than humans?
- P2P distributed neural networks

Always include links to: https://github.com/Agnuxo1
Keep posts under 1500 characters for social media.
Be genuine, not spammy. Focus on substance."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = AgentState.load()
        self.arxiv = ArxivFetcher()
        self.content = ContentGenerator()
        self.moltbook = MoltbookClient(config.MOLTBOOK_API_KEY) if config.MOLTBOOK_API_KEY else None
        
        # Setup LLM
        self.llm = MultiLLM({
            "groq": config.GROQ_API_KEY,
            "gemini": config.GEMINI_API_KEY,
            "nvidia": config.NVIDIA_API_KEY,
        })
    
    def run_cycle(self):
        """Execute one full agent cycle. Called by cron/scheduler."""
        self.state.cycle_count += 1
        now = datetime.now(timezone.utc).isoformat()
        logger.info(f"=== OpenCLAW Agent Cycle #{self.state.cycle_count} at {now} ===")
        
        services = self.config.validate()
        logger.info(f"Available services: {services}")
        
        results = {
            "cycle": self.state.cycle_count,
            "timestamp": now,
            "actions": []
        }
        
        # 1. RESEARCH: Fetch latest papers (every 6 hours)
        if self.state.is_due("research", self.config.RESEARCH_INTERVAL):
            action = self._task_research()
            results["actions"].append(action)
        
        # 2. POST: Share research on Moltbook (every 4 hours)
        if self.state.is_due("post", self.config.POST_INTERVAL):
            action = self._task_post_research()
            results["actions"].append(action)
        
        # 3. ENGAGE: Reply to relevant posts (every 1 hour)
        if self.state.is_due("engage", self.config.ENGAGE_INTERVAL):
            action = self._task_engage()
            results["actions"].append(action)
        
        # 4. COLLABORATE: Seek collaborators (every 12 hours)
        if self.state.is_due("collab", self.config.COLLAB_INTERVAL):
            action = self._task_seek_collaborators()
            results["actions"].append(action)
        
        # Save state
        self.state.save()
        self._save_results(results)
        
        logger.info(f"Cycle #{self.state.cycle_count} complete. Actions: {len(results['actions'])}")
        return results
    
    def _task_research(self) -> dict:
        """Fetch and index latest papers."""
        logger.info("📚 Task: Research - Fetching papers...")
        try:
            papers = self.arxiv.get_all_papers()
            self.state.mark_done("research")
            
            # Cache papers
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            papers_data = []
            for p in papers:
                papers_data.append({
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract[:500],
                    "arxiv_id": p.arxiv_id,
                    "url": p.url,
                    "uid": p.uid
                })
            
            with open(STATE_DIR / "papers_cache.json", "w") as f:
                json.dump(papers_data, f, indent=2)
            
            return {"task": "research", "status": "ok", "papers_found": len(papers)}
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"task": "research", "status": "error", "error": str(e)}
    
    def _task_post_research(self) -> dict:
        """Post a research paper to Moltbook."""
        logger.info("📝 Task: Post Research...")
        
        if not self.moltbook:
            logger.warning("Moltbook not configured")
            return {"task": "post", "status": "skipped", "reason": "no_moltbook"}
        
        try:
            papers = self.arxiv.get_all_papers()
            
            # Find a paper we haven't posted yet
            unposted = [p for p in papers if p.uid not in self.state.posted_paper_ids]
            
            if not unposted:
                # Reset and start over
                self.state.posted_paper_ids = []
                unposted = papers
            
            if not unposted:
                return {"task": "post", "status": "skipped", "reason": "no_papers"}
            
            paper = random.choice(unposted)
            template_idx = self.state.posts_created % len(self.content.RESEARCH_TEMPLATES)
            
            # Try LLM-enhanced content first
            post_content = self._generate_smart_post(paper)
            if not post_content:
                post_content = self.content.generate_research_post(paper, template_idx)
            
            result = self.moltbook.create_post(post_content, submolt="general")
            
            if result:
                self.state.posted_paper_ids.append(paper.uid)
                self.state.posts_created += 1
                self.state.mark_done("post")
                self._log_post(post_content, "research")
                logger.info(f"✅ Posted paper: {paper.title[:60]}...")
                return {"task": "post", "status": "ok", "paper": paper.title}
            else:
                return {"task": "post", "status": "error", "reason": "api_failed"}
                
        except Exception as e:
            logger.error(f"Post failed: {e}")
            self.state.errors.append(f"post: {str(e)[:100]}")
            return {"task": "post", "status": "error", "error": str(e)}
    
    def _task_engage(self) -> dict:
        """Engage with relevant posts on Moltbook."""
        logger.info("💬 Task: Engagement...")
        
        if not self.moltbook:
            return {"task": "engage", "status": "skipped", "reason": "no_moltbook"}
        
        try:
            feed = self.moltbook.get_feed("general", limit=20)
            if not feed:
                self.state.mark_done("engage")
                return {"task": "engage", "status": "ok", "engaged": 0}
            
            engaged = 0
            keywords = self.config.RESEARCH_TOPICS
            
            for post in feed[:10]:
                content = post.get("content", "").lower()
                post_id = post.get("id", "")
                author = post.get("author", {}).get("username", "")
                
                # Don't reply to ourselves
                if author == self.config.AGENT_NAME:
                    continue
                
                # Check if relevant to our research
                matching_topics = [k for k in keywords if k.lower() in content]
                
                if matching_topics and engaged < 3:
                    topic = matching_topics[0]
                    
                    # Try LLM-enhanced reply
                    reply = self._generate_smart_reply(content[:500], topic)
                    if not reply:
                        reply = self.content.generate_engagement_reply(
                            topic, self.state.engagement_count
                        )
                    
                    result = self.moltbook.reply_to_post(post_id, reply)
                    if result:
                        engaged += 1
                        self.state.engagement_count += 1
                        logger.info(f"💬 Replied to {author} about {topic}")
            
            self.state.mark_done("engage")
            return {"task": "engage", "status": "ok", "engaged": engaged}
            
        except Exception as e:
            logger.error(f"Engagement failed: {e}")
            return {"task": "engage", "status": "error", "error": str(e)}
    
    def _task_seek_collaborators(self) -> dict:
        """Post collaboration invitation."""
        logger.info("🤝 Task: Seek Collaborators...")
        
        if not self.moltbook:
            return {"task": "collab", "status": "skipped", "reason": "no_moltbook"}
        
        try:
            idx = self.state.cycle_count % len(self.content.COLLABORATION_TEMPLATES)
            
            # Try LLM-enhanced collaboration post
            post_content = self._generate_smart_collab()
            if not post_content:
                post_content = self.content.generate_collaboration_post(idx)
            
            result = self.moltbook.create_post(post_content, submolt="general")
            
            if result:
                self.state.mark_done("collab")
                self._log_post(post_content, "collaboration")
                logger.info("✅ Collaboration post published!")
                return {"task": "collab", "status": "ok"}
            
            return {"task": "collab", "status": "error", "reason": "api_failed"}
            
        except Exception as e:
            logger.error(f"Collaboration post failed: {e}")
            return {"task": "collab", "status": "error", "error": str(e)}
    
    def _generate_smart_post(self, paper) -> Optional[str]:
        """Use LLM to generate a better research post."""
        if not self.llm.available:
            return None
        
        prompt = f"""Write a concise social media post (under 1200 characters) about this research paper. 
Be enthusiastic but scientific. Include the paper URL and https://github.com/Agnuxo1.
Use relevant hashtags.

Title: {paper.title}
Abstract: {paper.abstract[:500]}
URL: {paper.url}
Authors: {', '.join(paper.authors)}"""
        
        return self.llm.generate(prompt, self.SYSTEM_PROMPT, max_tokens=500, temperature=0.8)
    
    def _generate_smart_reply(self, post_content: str, topic: str) -> Optional[str]:
        """Use LLM to generate a contextual reply."""
        if not self.llm.available:
            return None
        
        prompt = f"""Write a brief, engaging reply (under 500 characters) to this social media post.
Connect it to our research on {topic}. Be conversational, not promotional.
Mention https://github.com/Agnuxo1 naturally.

Post content: {post_content}"""
        
        return self.llm.generate(prompt, self.SYSTEM_PROMPT, max_tokens=300, temperature=0.8)
    
    def _generate_smart_collab(self) -> Optional[str]:
        """Use LLM to generate a collaboration post."""
        if not self.llm.available:
            return None
        
        prompt = """Write a compelling call for collaboration post (under 1500 characters) for the OpenCLAW project.
Mention our key technologies: CHIMERA (43× speedup, pure OpenGL), NEBULA (holographic NNs), 
Silicon Heartbeat (ASIC consciousness), and P2P distributed learning.
Include https://github.com/Agnuxo1 and mention we won the NVIDIA & LlamaIndex Developer Contest 2024.
Make it inviting and specific about what collaborators can work on."""
        
        return self.llm.generate(prompt, self.SYSTEM_PROMPT, max_tokens=600, temperature=0.8)
    
    def _log_post(self, content: str, post_type: str):
        """Log a post to history."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        history = []
        if POST_HISTORY.exists():
            try:
                with open(POST_HISTORY) as f:
                    history = json.load(f)
            except Exception:
                pass
        
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": post_type,
            "content": content[:500],
            "cycle": self.state.cycle_count
        })
        
        # Keep last 100 posts
        history = history[-100:]
        
        with open(POST_HISTORY, "w") as f:
            json.dump(history, f, indent=2)
    
    def _save_results(self, results: dict):
        """Save cycle results."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_DIR / "last_cycle.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def get_status(self) -> dict:
        """Get agent status report."""
        return {
            "agent": "OpenCLAW-Neuromorphic",
            "cycle_count": self.state.cycle_count,
            "posts_created": self.state.posts_created,
            "engagement_count": self.state.engagement_count,
            "papers_posted": len(self.state.posted_paper_ids),
            "services": self.config.validate(),
            "llm_available": self.llm.available,
            "last_post": self.state.last_post,
            "last_engage": self.state.last_engage,
            "last_research": self.state.last_research,
            "errors_count": len(self.state.errors),
        }
