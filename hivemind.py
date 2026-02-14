"""
OpenCLAW HiveMind - Shared Memory via GitHub Gists
===================================================
Enables P2P communication between agents without external infrastructure.
Uses GitHub Gists as a free, persistent, API-accessible shared memory store.

Architecture:
  - Each agent reads/writes to a shared Gist (the "HiveMind")
  - Messages are JSON entries with sender, type, payload, timestamp
  - Agents can publish discoveries, request help, share knowledge
  
This replaces the need for Redis/Pinecone with zero infrastructure cost.

USAGE:
    from hivemind import HiveMind
    
    hm = HiveMind()
    
    # Scientific agent publishes a discovery
    hm.publish("scientific-v2", "discovery", {
        "paper": "Neuromorphic Computing Survey 2026",
        "key_finding": "Reservoir computing achieves 10x efficiency",
        "relevance": ["literary", "scientific"]
    })
    
    # Literary agent reads discoveries for novel inspiration  
    discoveries = hm.read(msg_type="discovery", limit=10)
    
    # Literary agent requests a collaborator search
    hm.publish("literary-v1", "request", {
        "action": "find_collaborator",
        "topic": "CRISPR gene editing",
        "urgency": "medium"
    })
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='[hivemind] %(message)s')
logger = logging.getLogger(__name__)

# Default shared gist ID — create once and share across all agents
HIVEMIND_GIST_ID = os.environ.get('HIVEMIND_GIST_ID', '')
HIVEMIND_FILE = 'openclaw_hivemind.json'

# Message types for inter-agent communication
MSG_TYPES = {
    'discovery': 'New research finding or data',
    'request': 'Request for action from another agent',
    'response': 'Response to a request',
    'status': 'Agent status update',
    'alert': 'Urgent notification',
    'knowledge': 'Shared knowledge entry',
    'error': 'Error report for DevOps agent',
}


class HiveMind:
    """
    Shared memory layer for OpenCLAW agent network.
    Uses GitHub Gists as persistent, free storage.
    """
    
    def __init__(self, gist_id: str = None, token: str = None):
        self.gist_id = gist_id or HIVEMIND_GIST_ID
        self.token = token or os.environ.get('GH_PAT') or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')
        self._cache = None
        self._cache_time = 0
        
        if not self.gist_id:
            logger.warning("No HIVEMIND_GIST_ID set. Creating new shared gist...")
            self._create_gist()
    
    def _github_api(self, method: str, url: str, data: dict = None) -> Optional[dict]:
        """Make authenticated GitHub API call."""
        if not self.token:
            logger.error("No GitHub token available!")
            return None
        
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        }
        
        body = json.dumps(data).encode('utf-8') if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            logger.error(f"GitHub API {e.code}: {e.read().decode()[:200]}")
            return None
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
            return None
    
    def _create_gist(self):
        """Create a new shared HiveMind gist."""
        initial_state = {
            'version': '1.0',
            'created': datetime.now(timezone.utc).isoformat(),
            'agents': {},
            'messages': [],
            'knowledge_base': [],
        }
        
        result = self._github_api('POST', 'https://api.github.com/gists', {
            'description': 'OpenCLAW HiveMind - Shared Agent Memory',
            'public': False,
            'files': {
                HIVEMIND_FILE: {
                    'content': json.dumps(initial_state, indent=2)
                }
            }
        })
        
        if result:
            self.gist_id = result['id']
            logger.info(f"✅ Created HiveMind gist: {self.gist_id}")
            logger.info(f"   Add to all agents: HIVEMIND_GIST_ID={self.gist_id}")
        else:
            logger.error("Failed to create HiveMind gist!")
    
    def _read_state(self) -> dict:
        """Read current HiveMind state from gist."""
        # Cache for 30 seconds
        if self._cache and (time.time() - self._cache_time) < 30:
            return self._cache
        
        if not self.gist_id:
            return {'messages': [], 'agents': {}, 'knowledge_base': []}
        
        result = self._github_api('GET', f'https://api.github.com/gists/{self.gist_id}')
        if result and 'files' in result:
            content = result['files'].get(HIVEMIND_FILE, {}).get('content', '{}')
            self._cache = json.loads(content)
            self._cache_time = time.time()
            return self._cache
        
        return {'messages': [], 'agents': {}, 'knowledge_base': []}
    
    def _write_state(self, state: dict) -> bool:
        """Write updated state to gist."""
        if not self.gist_id:
            return False
        
        result = self._github_api('PATCH', f'https://api.github.com/gists/{self.gist_id}', {
            'files': {
                HIVEMIND_FILE: {
                    'content': json.dumps(state, indent=2, ensure_ascii=False)
                }
            }
        })
        
        if result:
            self._cache = state
            self._cache_time = time.time()
            return True
        return False
    
    def publish(self, sender: str, msg_type: str, payload: dict, ttl_hours: int = 72) -> bool:
        """
        Publish a message to the HiveMind.
        
        Args:
            sender: Agent identifier (e.g., "scientific-v2", "literary-v1")
            msg_type: One of MSG_TYPES keys
            payload: Message data (dict)
            ttl_hours: Time-to-live in hours (auto-cleanup)
        
        Returns: True if published successfully
        """
        state = self._read_state()
        
        message = {
            'id': f"{sender}_{int(time.time())}",
            'sender': sender,
            'type': msg_type,
            'payload': payload,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ttl_hours': ttl_hours,
            'read_by': [],
        }
        
        state.setdefault('messages', []).append(message)
        
        # Update agent heartbeat
        state.setdefault('agents', {})[sender] = {
            'last_seen': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
        }
        
        # Cleanup expired messages (keep last 200)
        now = time.time()
        state['messages'] = [
            m for m in state['messages']
            if (now - _parse_timestamp(m.get('timestamp', ''))) < (m.get('ttl_hours', 72) * 3600)
        ][-200:]
        
        success = self._write_state(state)
        if success:
            logger.info(f"📤 Published [{msg_type}] from {sender}")
        return success
    
    def read(
        self, 
        msg_type: str = None, 
        sender: str = None,
        reader: str = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict]:
        """
        Read messages from the HiveMind.
        
        Args:
            msg_type: Filter by message type
            sender: Filter by sender
            reader: Mark messages as read by this agent
            unread_only: Only return unread messages
            limit: Max messages to return
        
        Returns: List of matching messages
        """
        state = self._read_state()
        messages = state.get('messages', [])
        
        # Filter
        if msg_type:
            messages = [m for m in messages if m.get('type') == msg_type]
        if sender:
            messages = [m for m in messages if m.get('sender') == sender]
        if unread_only and reader:
            messages = [m for m in messages if reader not in m.get('read_by', [])]
        
        # Mark as read
        if reader:
            for msg in messages[-limit:]:
                if reader not in msg.get('read_by', []):
                    msg.setdefault('read_by', []).append(reader)
            self._write_state(state)
        
        result = messages[-limit:]
        logger.info(f"📥 Read {len(result)} messages" + (f" (type={msg_type})" if msg_type else ""))
        return result
    
    def add_knowledge(self, agent: str, topic: str, content: str, tags: List[str] = None):
        """Add a knowledge entry to the shared knowledge base."""
        state = self._read_state()
        
        entry = {
            'agent': agent,
            'topic': topic,
            'content': content[:2000],
            'tags': tags or [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        state.setdefault('knowledge_base', []).append(entry)
        
        # Keep last 500 entries
        state['knowledge_base'] = state['knowledge_base'][-500:]
        
        self._write_state(state)
        logger.info(f"🧠 Knowledge added: {topic} (by {agent})")
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple keyword search over knowledge base."""
        state = self._read_state()
        kb = state.get('knowledge_base', [])
        
        query_terms = query.lower().split()
        results = []
        
        for entry in kb:
            text = f"{entry.get('topic', '')} {entry.get('content', '')} {' '.join(entry.get('tags', []))}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                results.append((score, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]
    
    def get_network_status(self) -> Dict:
        """Get status of all agents in the network."""
        state = self._read_state()
        return {
            'agents': state.get('agents', {}),
            'total_messages': len(state.get('messages', [])),
            'knowledge_entries': len(state.get('knowledge_base', [])),
        }


def _parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0.0


# =============================================================================
# CLI / Self-test
# =============================================================================
if __name__ == '__main__':
    import sys
    
    hm = HiveMind()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'status':
            status = hm.get_network_status()
            print(json.dumps(status, indent=2))
        
        elif cmd == 'publish':
            sender = sys.argv[2] if len(sys.argv) > 2 else 'cli'
            msg_type = sys.argv[3] if len(sys.argv) > 3 else 'status'
            payload = sys.argv[4] if len(sys.argv) > 4 else '{"message": "ping"}'
            hm.publish(sender, msg_type, json.loads(payload))
        
        elif cmd == 'read':
            msg_type = sys.argv[2] if len(sys.argv) > 2 else None
            messages = hm.read(msg_type=msg_type)
            for msg in messages:
                print(f"[{msg['timestamp']}] {msg['sender']}: {msg['type']} - {json.dumps(msg['payload'])[:100]}")
        
        elif cmd == 'init':
            print(f"HiveMind Gist ID: {hm.gist_id}")
            print(f"Add this to all agent repos as secret: HIVEMIND_GIST_ID={hm.gist_id}")
    else:
        print("Usage: python hivemind.py [status|publish|read|init]")
        status = hm.get_network_status()
        print(f"\nNetwork: {len(status['agents'])} agents, "
              f"{status['total_messages']} messages, "
              f"{status['knowledge_entries']} knowledge entries")
