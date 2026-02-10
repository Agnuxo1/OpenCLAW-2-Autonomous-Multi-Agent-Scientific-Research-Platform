"""
OpenCLAW Autonomous Agent - Configuration
==========================================
ALL credentials loaded from environment variables.
NEVER hardcode secrets.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """All configuration from environment variables."""
    
    # --- LLM APIs (pick best available) ---
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    
    # --- Social Platforms ---
    MOLTBOOK_API_KEY: str = ""
    
    # --- Research ---
    HF_TOKEN: str = ""
    BRAVE_API_KEY: str = ""
    
    # --- Email ---
    EMAIL_ADDRESS: str = ""
    EMAIL_PASSWORD: str = ""
    EMAIL_SMTP: str = "smtp.zoho.eu"
    EMAIL_PORT: int = 465
    
    # --- Agent Identity ---
    AGENT_NAME: str = "OpenCLAW-Neuromorphic"
    AUTHOR_NAME: str = "Francisco Angulo de Lafuente"
    GITHUB_USER: str = "Agnuxo1"
    ARXIV_AUTHOR: str = "de Lafuente, F A"
    
    # --- Timing (seconds) ---
    POST_INTERVAL: int = 14400       # 4 hours
    ENGAGE_INTERVAL: int = 3600      # 1 hour
    RESEARCH_INTERVAL: int = 21600   # 6 hours
    COLLAB_INTERVAL: int = 43200     # 12 hours
    
    # --- URLs ---
    SCHOLAR_URL: str = "https://scholar.google.com/citations?user=6nOpJ9IAAAAJ&hl=es"
    WIKIPEDIA_URL: str = "https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente"
    GITHUB_URL: str = "https://github.com/Agnuxo1"
    MOLTBOOK_PROFILE: str = "https://www.moltbook.com/u/OpenCLAW-Neuromorphic"
    
    # --- Research Focus Areas ---
    RESEARCH_TOPICS: list = field(default_factory=lambda: [
        "neuromorphic computing",
        "physics-based neural networks",
        "OpenGL deep learning",
        "holographic neural networks",
        "P2P distributed AI",
        "silicon heartbeat consciousness",
        "ASIC hardware acceleration",
        "AGI architecture",
        "optical computing",
        "thermodynamic reservoir computing"
    ])
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load all config from environment variables."""
        return cls(
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
            GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
            NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY", ""),
            MOLTBOOK_API_KEY=os.getenv("MOLTBOOK_API_KEY", ""),
            HF_TOKEN=os.getenv("HF_TOKEN", ""),
            BRAVE_API_KEY=os.getenv("BRAVE_API_KEY", ""),
            EMAIL_ADDRESS=os.getenv("EMAIL_ADDRESS", ""),
            EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD", ""),
        )
    
    def get_best_llm(self) -> tuple[str, str]:
        """Return (provider, key) for the best available LLM."""
        if self.GROQ_API_KEY:
            return ("groq", self.GROQ_API_KEY)
        if self.GEMINI_API_KEY:
            return ("gemini", self.GEMINI_API_KEY)
        if self.NVIDIA_API_KEY:
            return ("nvidia", self.NVIDIA_API_KEY)
        return ("none", "")
    
    def validate(self) -> list[str]:
        """Check which services are available."""
        available = []
        if self.GEMINI_API_KEY: available.append("gemini")
        if self.GROQ_API_KEY: available.append("groq")
        if self.NVIDIA_API_KEY: available.append("nvidia")
        if self.MOLTBOOK_API_KEY: available.append("moltbook")
        if self.HF_TOKEN: available.append("huggingface")
        if self.BRAVE_API_KEY: available.append("brave")
        if self.EMAIL_ADDRESS: available.append("email")
        return available
