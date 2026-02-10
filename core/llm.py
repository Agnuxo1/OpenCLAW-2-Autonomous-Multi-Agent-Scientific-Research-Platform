"""
LLM Connector - Multi-Provider Intelligence
=============================================
Connects to available LLM APIs for content generation and reasoning.
Falls back gracefully between providers.
"""
import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger("openclaw.llm")


class LLMConnector:
    """Multi-provider LLM connector."""
    
    PROVIDERS = {
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "llama-3.3-70b-versatile",
            "header_key": "Authorization",
            "header_prefix": "Bearer ",
        },
        "gemini": {
            "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            "model": "gemini-2.0-flash",
            "header_key": "x-goog-api-key",
            "header_prefix": "",
        },
        "nvidia": {
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "model": "meta/llama-3.1-70b-instruct",
            "header_key": "Authorization",
            "header_prefix": "Bearer ",
        },
    }
    
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.config = self.PROVIDERS.get(provider, {})
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> Optional[str]:
        """Generate text using the configured LLM."""
        if not self.api_key or not self.config:
            logger.warning(f"LLM provider '{self.provider}' not configured")
            return None
        
        try:
            if self.provider == "gemini":
                return self._generate_gemini(prompt, system, max_tokens, temperature)
            else:
                return self._generate_openai_compat(prompt, system, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            return None
    
    def _generate_openai_compat(self, prompt: str, system: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate using OpenAI-compatible API (Groq, NVIDIA)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        data = json.dumps({
            "model": self.config["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        
        headers = {
            "Content-Type": "application/json",
            self.config["header_key"]: f"{self.config['header_prefix']}{self.api_key}",
        }
        
        req = urllib.request.Request(self.config["url"], data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        
        return result["choices"][0]["message"]["content"]
    
    def _generate_gemini(self, prompt: str, system: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate using Google Gemini API."""
        url = f"{self.config['url']}?key={self.api_key}"
        
        parts = []
        if system:
            parts.append({"text": f"System: {system}\n\nUser: {prompt}"})
        else:
            parts.append({"text": prompt})
        
        data = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }).encode()
        
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        
        return result["candidates"][0]["content"]["parts"][0]["text"]


class MultiLLM:
    """Try multiple LLM providers in order, with key rotation."""
    
    def __init__(self, providers: dict[str, str]):
        """providers: dict of {provider_name: api_key} or {provider_name: 'key1,key2,key3'}"""
        self.connectors = []
        # Priority order: nvidia (working), groq (fast), gemini (free)
        for name in ["nvidia", "groq", "gemini"]:
            if name in providers and providers[name]:
                # Support comma-separated multiple keys
                keys = [k.strip() for k in providers[name].split(",") if k.strip()]
                for key in keys:
                    self.connectors.append(LLMConnector(name, key))
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Try each provider until one works."""
        for connector in self.connectors:
            try:
                result = connector.generate(prompt, system, max_tokens, temperature)
                if result:
                    logger.info(f"LLM response from {connector.provider}")
                    return result
            except Exception as e:
                logger.warning(f"Provider {connector.provider} failed: {e}")
                continue
        
        logger.warning("All LLM providers failed, using template")
        return ""
    
    @property
    def available(self) -> bool:
        return len(self.connectors) > 0
