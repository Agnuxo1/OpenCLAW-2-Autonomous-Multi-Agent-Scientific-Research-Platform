"""
OpenCLAW Unified LLM Provider
==============================
Centralizes ALL 29 API keys with automatic failover.
Supports both CSV format (Literary v1) and numbered format (Literary2 GLM5).

USAGE:
    from unified_llm import UnifiedLLM
    llm = UnifiedLLM()
    response = llm.generate("Write a blog post about...")

ENVIRONMENT VARIABLES (supports both naming conventions):
    CSV format:     GROQ_API_KEYS="key1,key2,key3"
    Numbered format: GROQ_API_KEY_1="key1", GROQ_API_KEY_2="key2"
    Single format:   GROQ_API_KEY="key1"
    
Drop this file into ANY agent repo to unify LLM access.
"""

import os
import json
import time
import random
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='[llm] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Provider Configuration
# =============================================================================
PROVIDERS = {
    'groq': {
        'base_url': 'https://api.groq.com/openai/v1/chat/completions',
        'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
        'env_keys': ['GROQ_API_KEYS', 'GROQ_API_KEY'],
        'env_numbered_prefix': 'GROQ_API_KEY_',
        'max_numbered': 10,
        'rpm': 30,
    },
    'nvidia': {
        'base_url': 'https://integrate.api.nvidia.com/v1/chat/completions',
        'models': ['meta/llama-3.1-405b-instruct', 'meta/llama-3.1-70b-instruct'],
        'env_keys': ['NVIDIA_API_KEYS', 'NVIDIA_API_KEY'],
        'env_numbered_prefix': 'NVIDIA_API_KEY_',
        'max_numbered': 5,
        'rpm': 20,
    },
    'openrouter': {
        'base_url': 'https://openrouter.ai/api/v1/chat/completions',
        'models': ['meta-llama/llama-3.3-70b-instruct:free', 'google/gemma-2-9b-it:free'],
        'env_keys': ['OPENROUTER_API_KEYS', 'OPENROUTER_API_KEY'],
        'env_numbered_prefix': 'OPENROUTER_API_KEY_',
        'max_numbered': 10,
        'rpm': 20,
    },
    'mistral': {
        'base_url': 'https://api.mistral.ai/v1/chat/completions',
        'models': ['mistral-small-latest', 'open-mistral-7b'],
        'env_keys': ['MISTRAL_API_KEYS', 'MISTRAL_API_KEY'],
        'env_numbered_prefix': 'MISTRAL_API_KEY_',
        'max_numbered': 5,
        'rpm': 15,
    },
    'deepseek': {
        'base_url': 'https://api.deepseek.com/v1/chat/completions',
        'models': ['deepseek-chat', 'deepseek-reasoner'],
        'env_keys': ['DEEPSEEK_API_KEYS', 'DEEPSEEK_API_KEY'],
        'env_numbered_prefix': 'DEEPSEEK_API_KEY_',
        'max_numbered': 10,
        'rpm': 10,
    },
    'zhipuai': {
        'base_url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
        'models': ['glm-4-flash', 'glm-4'],
        'env_keys': ['ZHIPUAI_API_KEYS', 'ZHIPUAI_API_KEY', 'GLM_API_KEY'],
        'env_numbered_prefix': 'ZHIPUAI_API_KEY_',
        'max_numbered': 10,
        'rpm': 10,
    },
    'gemini': {
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        'models': ['gemini-1.5-flash', 'gemini-1.5-pro'],
        'env_keys': ['GEMINI_API_KEYS', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'env_numbered_prefix': 'GEMINI_API_KEY_',
        'max_numbered': 10,
        'rpm': 15,
        'custom_format': True,  # Gemini uses different API format
    },
}


class ProviderState:
    """Track state for a single API key."""
    def __init__(self, provider: str, key: str, index: int):
        self.provider = provider
        self.key = key
        self.index = index
        self.failures = 0
        self.last_used = 0.0
        self.disabled = False
        self.disable_until = 0.0
    
    @property
    def available(self) -> bool:
        if self.disabled and time.time() < self.disable_until:
            return False
        if self.disabled and time.time() >= self.disable_until:
            self.disabled = False
            self.failures = 0
        return not self.disabled
    
    def mark_success(self):
        self.failures = 0
        self.last_used = time.time()
    
    def mark_failure(self, error_code: int = 0):
        self.failures += 1
        self.last_used = time.time()
        
        # Disable key temporarily based on error type
        if error_code == 401 or error_code == 403:
            # Invalid/expired key — disable for 1 hour
            self.disabled = True
            self.disable_until = time.time() + 3600
            logger.warning(f"  [{self.provider}#{self.index}] Key disabled (auth error {error_code})")
        elif error_code == 429:
            # Rate limited — back off exponentially
            backoff = min(600, 30 * (2 ** self.failures))
            self.disable_until = time.time() + backoff
            self.disabled = True
            logger.warning(f"  [{self.provider}#{self.index}] Rate limited, backoff {backoff}s")
        elif self.failures >= 3:
            # General failure — disable for 5 minutes
            self.disabled = True
            self.disable_until = time.time() + 300
            logger.warning(f"  [{self.provider}#{self.index}] 3+ failures, disabled 5min")


class UnifiedLLM:
    """
    Unified LLM client with 29-key pool and automatic failover.
    
    Usage:
        llm = UnifiedLLM()
        response = llm.generate("Hello world")
        
    The client automatically:
    - Loads all keys from environment (CSV, numbered, or single format)
    - Rotates between providers and keys
    - Handles rate limits with exponential backoff
    - Falls back to next provider on failure
    """
    
    def __init__(self, preferred_providers: List[str] = None):
        self.keys: List[ProviderState] = []
        self.preferred_providers = preferred_providers or list(PROVIDERS.keys())
        self._load_all_keys()
        
        if not self.keys:
            logger.error("⚠️ NO API KEYS FOUND in environment!")
            logger.error("Expected variables: GROQ_API_KEYS, NVIDIA_API_KEY, etc.")
            logger.error("Or numbered: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")
    
    def _load_all_keys(self):
        """Load keys from all supported formats."""
        total = 0
        
        for provider_name, config in PROVIDERS.items():
            keys_found = set()
            
            # Format 1: CSV (e.g., GROQ_API_KEYS="key1,key2,key3")
            for env_var in config['env_keys']:
                val = os.environ.get(env_var, '').strip()
                if val:
                    for k in val.split(','):
                        k = k.strip()
                        if k and len(k) > 10:
                            keys_found.add(k)
            
            # Format 2: Numbered (e.g., GROQ_API_KEY_1, GROQ_API_KEY_2)
            prefix = config['env_numbered_prefix']
            for i in range(1, config['max_numbered'] + 1):
                val = os.environ.get(f"{prefix}{i}", '').strip()
                if val and len(val) > 10:
                    keys_found.add(val)
            
            # Create ProviderState for each key
            for idx, key in enumerate(keys_found):
                self.keys.append(ProviderState(provider_name, key, idx + 1))
                total += 1
            
            if keys_found:
                logger.info(f"  {provider_name}: {len(keys_found)} key(s) loaded")
        
        # Shuffle to distribute load
        random.shuffle(self.keys)
        logger.info(f"✅ Total API keys loaded: {total} across {len(set(k.provider for k in self.keys))} providers")
    
    def _get_available_keys(self) -> List[ProviderState]:
        """Get available keys sorted by preference and freshness."""
        available = [k for k in self.keys if k.available]
        
        # Sort: preferred providers first, then least recently used
        def sort_key(k):
            pref_idx = self.preferred_providers.index(k.provider) if k.provider in self.preferred_providers else 99
            return (pref_idx, k.failures, k.last_used)
        
        return sorted(available, key=sort_key)
    
    def _call_openai_compatible(
        self, 
        provider: str, 
        key: str, 
        model: str, 
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Call OpenAI-compatible API (Groq, NVIDIA, OpenRouter, Mistral, DeepSeek)."""
        config = PROVIDERS[provider]
        url = config['base_url']
        
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}',
        }
        
        # OpenRouter needs extra headers
        if provider == 'openrouter':
            headers['HTTP-Referer'] = 'https://github.com/Agnuxo1/OpenCLAW'
            headers['X-Title'] = 'OpenCLAW Agent'
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    
    def _call_gemini(
        self,
        key: str,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Call Gemini API (different format)."""
        url = PROVIDERS['gemini']['base_url'].format(model=model) + f'?key={key}'
        
        # Convert OpenAI messages to Gemini format
        contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append({
                'role': role,
                'parts': [{'text': msg['content']}]
            })
        
        payload = {
            'contents': contents,
            'generationConfig': {
                'maxOutputTokens': max_tokens,
                'temperature': temperature,
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text']
    
    def _call_zhipuai(
        self,
        key: str,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Call ZhipuAI/GLM API."""
        url = PROVIDERS['zhipuai']['base_url']
        
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}',
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    
    def generate(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Generate text using the best available provider.
        Automatically handles failover across all 29 keys.
        """
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ]
        
        available = self._get_available_keys()
        
        if not available:
            logger.error("❌ All API keys exhausted or rate-limited!")
            return None
        
        for key_state in available:
            provider = key_state.provider
            config = PROVIDERS[provider]
            model = config['models'][0]  # Use primary model
            
            try:
                logger.info(f"Trying {provider}#{key_state.index} ({model})...")
                
                if provider == 'gemini':
                    result = self._call_gemini(key_state.key, model, messages, max_tokens, temperature)
                elif provider == 'zhipuai':
                    result = self._call_zhipuai(key_state.key, model, messages, max_tokens, temperature)
                else:
                    result = self._call_openai_compatible(provider, key_state.key, model, messages, max_tokens, temperature)
                
                if result:
                    key_state.mark_success()
                    logger.info(f"✅ Success via {provider}#{key_state.index}")
                    return result
                
            except urllib.error.HTTPError as e:
                key_state.mark_failure(e.code)
                logger.warning(f"  {provider}#{key_state.index} HTTP {e.code}")
                continue
                
            except Exception as e:
                key_state.mark_failure()
                logger.warning(f"  {provider}#{key_state.index} error: {e}")
                continue
        
        logger.error("❌ ALL providers failed for this request")
        return None
    
    @property
    def status(self) -> Dict:
        """Return status of all providers."""
        status = {}
        for key_state in self.keys:
            if key_state.provider not in status:
                status[key_state.provider] = {'total': 0, 'available': 0, 'disabled': 0}
            status[key_state.provider]['total'] += 1
            if key_state.available:
                status[key_state.provider]['available'] += 1
            else:
                status[key_state.provider]['disabled'] += 1
        return status


# =============================================================================
# Quick test
# =============================================================================
if __name__ == '__main__':
    llm = UnifiedLLM()
    
    print("\n📊 Provider Status:")
    for provider, stats in llm.status.items():
        print(f"  {provider}: {stats['available']}/{stats['total']} available")
    
    print("\n🧪 Testing generation...")
    result = llm.generate("Say 'Hello, OpenCLAW!' in one sentence.")
    if result:
        print(f"\n✅ Response: {result[:200]}")
    else:
        print("\n❌ All providers failed. Check your API keys.")
