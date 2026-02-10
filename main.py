#!/usr/bin/env python3
"""
OpenCLAW Autonomous Multi-Agent Scientific Research Platform
=============================================================
Main entry point.

Usage:
    python main.py run          # Run one cycle (for cron/GitHub Actions)
    python main.py status       # Show agent status
    python main.py daemon       # Run continuously (for server deployment)
    python main.py test         # Test configuration without posting
"""
import sys
import os
import time
import logging
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.agent import OpenCLAWAgent


def setup_logging():
    """Configure logging."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Also log to file if state dir exists
    state_dir = os.getenv("STATE_DIR", "state")
    os.makedirs(state_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(state_dir, "agent.log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def cmd_run():
    """Run one agent cycle."""
    config = Config.from_env()
    agent = OpenCLAWAgent(config)
    
    print(f"\n🤖 OpenCLAW Agent - Cycle Start")
    print(f"   Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"   Services: {config.validate()}")
    print()
    
    results = agent.run_cycle()
    
    print(f"\n📊 Cycle Results:")
    for action in results.get("actions", []):
        status = "✅" if action.get("status") == "ok" else "⚠️"
        print(f"   {status} {action.get('task', '?')}: {action.get('status', '?')}")
    
    if not results.get("actions"):
        print("   ℹ️  No tasks due this cycle")
    
    print()
    return 0


def cmd_status():
    """Show agent status."""
    config = Config.from_env()
    agent = OpenCLAWAgent(config)
    status = agent.get_status()
    
    print(f"\n🤖 OpenCLAW Agent Status")
    print(f"   {'='*40}")
    for key, value in status.items():
        print(f"   {key}: {value}")
    print()
    return 0


def cmd_daemon():
    """Run continuously with sleep between cycles."""
    config = Config.from_env()
    interval = int(os.getenv("DAEMON_INTERVAL", "1800"))  # 30 min default
    
    print(f"\n🤖 OpenCLAW Agent - Daemon Mode")
    print(f"   Interval: {interval}s ({interval//60} min)")
    print(f"   Services: {config.validate()}")
    print(f"   Press Ctrl+C to stop\n")
    
    while True:
        try:
            agent = OpenCLAWAgent(config)
            results = agent.run_cycle()
            
            actions = len(results.get("actions", []))
            print(f"   [{datetime.now(timezone.utc).strftime('%H:%M')}] "
                  f"Cycle #{results['cycle']} - {actions} actions")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Agent stopped by user")
            return 0
        except Exception as e:
            logging.error(f"Daemon cycle error: {e}")
            print(f"   ⚠️ Error: {e}")
        
        time.sleep(interval)


def cmd_test():
    """Test configuration without making any posts."""
    config = Config.from_env()
    
    print(f"\n🧪 OpenCLAW Agent - Test Mode")
    print(f"   {'='*40}")
    
    # Check services
    services = config.validate()
    print(f"\n   Available services: {services}")
    
    # Test ArXiv
    from research.arxiv_fetcher import ArxivFetcher
    fetcher = ArxivFetcher()
    papers = fetcher.get_all_papers()
    print(f"\n   📚 Papers found: {len(papers)}")
    for p in papers[:3]:
        print(f"      - {p.title[:70]}...")
    
    # Test LLM
    from core.llm import MultiLLM
    llm = MultiLLM({
        "groq": config.GROQ_API_KEY,
        "gemini": config.GEMINI_API_KEY,
        "nvidia": config.NVIDIA_API_KEY,
    })
    if llm.available:
        print(f"\n   🧠 LLM available, testing...")
        response = llm.generate("Say 'OpenCLAW is online!' in exactly those words.", max_tokens=50)
        print(f"      Response: {response[:100] if response else 'FAILED'}")
    else:
        print(f"\n   ⚠️ No LLM configured")
    
    # Test Moltbook
    if config.MOLTBOOK_API_KEY:
        from social.moltbook import MoltbookClient
        mb = MoltbookClient(config.MOLTBOOK_API_KEY)
        print(f"\n   📱 Moltbook configured (not posting in test mode)")
    else:
        print(f"\n   ⚠️ Moltbook not configured")
    
    # Test content generation
    from social.moltbook import ContentGenerator
    cg = ContentGenerator()
    if papers:
        post = cg.generate_research_post(papers[0])
        print(f"\n   📝 Sample post ({len(post)} chars):")
        print(f"      {post[:200]}...")
    
    print(f"\n   ✅ Test complete!")
    return 0


def cmd_healthcheck():
    """Health check endpoint for monitoring."""
    print(json.dumps({
        "status": "healthy",
        "agent": "OpenCLAW-Neuromorphic",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))
    return 0


def main():
    setup_logging()
    
    if len(sys.argv) < 2:
        cmd = "run"
    else:
        cmd = sys.argv[1].lower()
    
    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "daemon": cmd_daemon,
        "test": cmd_test,
        "health": cmd_healthcheck,
    }
    
    if cmd in commands:
        return commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands.keys())}")
        return 1


if __name__ == "__main__":
    import json  # for healthcheck
    sys.exit(main() or 0)
