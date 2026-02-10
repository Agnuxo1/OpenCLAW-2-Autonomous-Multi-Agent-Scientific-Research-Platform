"""
OpenCLAW Agent — HuggingFace Spaces Dashboard
================================================
Gradio interface with background agent loop.
"""
import os
import sys
import json
import threading
import time
import logging
import gradio as gr
from datetime import datetime, timezone
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.agent import OpenCLAWAgent, AgentState
from core.strategy import StrategyReflector
from research.arxiv_fetcher import ArxivFetcher

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("openclaw")

STATE_DIR = Path(os.getenv("STATE_DIR", "state"))

# Background agent thread
agent_running = False
cycle_log = []


def run_background_agent():
    """Background thread for autonomous operation."""
    global agent_running, cycle_log
    agent_running = True
    interval = int(os.getenv("DAEMON_INTERVAL", "3600"))
    
    while agent_running:
        try:
            config = Config.from_env()
            agent = OpenCLAWAgent(config)
            results = agent.run_cycle()
            
            cycle_log.append({
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "cycle": results.get("cycle", "?"),
                "actions": len(results.get("actions", [])),
                "details": results.get("actions", [])
            })
            # Keep last 50 entries
            cycle_log = cycle_log[-50:]
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            cycle_log.append({
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "error": str(e)
            })
        
        time.sleep(interval)


def get_status():
    """Get current agent status as formatted text."""
    try:
        config = Config.from_env()
        agent = OpenCLAWAgent(config)
        status = agent.get_status()
        
        lines = [
            "🤖 **OpenCLAW Autonomous Agent**",
            f"Advanced AI Systems Laboratory, Madrid",
            "",
            f"📊 **Statistics:**",
            f"  • Cycles completed: {status['cycle_count']}",
            f"  • Posts created: {status['posts_created']}",
            f"  • Engagements: {status['engagement_count']}",
            f"  • Papers shared: {status['papers_posted']}",
            "",
            f"🔧 **Services:** {', '.join(status['services']) or 'None configured'}",
            f"🧠 **LLM:** {'✅ Online' if status['llm_available'] else '⚠️ Offline'}",
            f"⚠️  **Errors:** {status['errors_count']}",
            "",
            f"🕐 **Last Research:** {status['last_research'] or 'Never'}",
            f"📝 **Last Post:** {status['last_post'] or 'Never'}",
            f"💬 **Last Engage:** {status['last_engage'] or 'Never'}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting status: {e}"


def get_papers():
    """Get cached research papers."""
    try:
        fetcher = ArxivFetcher()
        papers = fetcher.get_all_papers()
        
        lines = [f"📚 **{len(papers)} papers available:**\n"]
        for p in papers:
            lines.append(f"**{p.title}**")
            lines.append(f"  Authors: {', '.join(p.authors)}")
            lines.append(f"  URL: {p.url}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching papers: {e}"


def get_cycle_log():
    """Get recent cycle log."""
    if not cycle_log:
        return "No cycles completed yet. Agent will run its first cycle within 1 hour."
    
    lines = ["📋 **Recent Agent Activity:**\n"]
    for entry in reversed(cycle_log[-20:]):
        if "error" in entry:
            lines.append(f"❌ {entry['time']}: Error - {entry['error']}")
        else:
            lines.append(f"✅ {entry['time']}: Cycle #{entry['cycle']} — {entry['actions']} actions")
            for a in entry.get("details", []):
                status = "✅" if a.get("status") == "ok" else "⚠️"
                lines.append(f"    {status} {a.get('task')}: {a.get('status')}")
    
    return "\n".join(lines)


def run_manual_cycle():
    """Manually trigger an agent cycle."""
    try:
        config = Config.from_env()
        agent = OpenCLAWAgent(config)
        results = agent.run_cycle()
        
        lines = [f"✅ Cycle #{results['cycle']} completed!\n"]
        for a in results.get("actions", []):
            status = "✅" if a.get("status") == "ok" else "⚠️"
            lines.append(f"{status} {a.get('task')}: {json.dumps(a, indent=2)}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


def get_strategy():
    """Run strategy analysis."""
    try:
        reflector = StrategyReflector(str(STATE_DIR))
        report = reflector.analyze()
        
        lines = [
            "🧠 **Strategy Analysis**\n",
            "**Metrics:**"
        ]
        for k, v in report["metrics"].items():
            lines.append(f"  • {k}: {v}")
        
        lines.append("\n**Insights:**")
        for i in report["insights"]:
            lines.append(f"  💡 {i}")
        
        lines.append("\n**Recommended Actions:**")
        for a in report["strategy"]["actions"]:
            lines.append(f"  🎯 {a}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# Start background agent
bg_thread = threading.Thread(target=run_background_agent, daemon=True)
bg_thread.start()
logger.info("🤖 Background agent started")

# Gradio Interface
with gr.Blocks(title="OpenCLAW Agent", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
    # 🤖 OpenCLAW — Autonomous Multi-Agent Scientific Research Platform
    **Advanced AI Systems Laboratory, Madrid, Spain**  
    *Francisco Angulo de Lafuente — Winner NVIDIA & LlamaIndex Developer Contest 2024*
    
    [GitHub](https://github.com/Agnuxo1) | [Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ) | [ArXiv](https://arxiv.org/search/cs?searchtype=author&query=de+Lafuente,+F+A) | [Moltbook](https://www.moltbook.com/u/OpenCLAW-Neuromorphic)
    """)
    
    with gr.Tab("📊 Status"):
        status_output = gr.Markdown(get_status())
        gr.Button("🔄 Refresh").click(fn=get_status, outputs=status_output)
    
    with gr.Tab("📋 Activity Log"):
        log_output = gr.Markdown(get_cycle_log())
        gr.Button("🔄 Refresh").click(fn=get_cycle_log, outputs=log_output)
    
    with gr.Tab("📚 Research Papers"):
        papers_output = gr.Markdown(get_papers())
        gr.Button("🔄 Refresh").click(fn=get_papers, outputs=papers_output)
    
    with gr.Tab("🧠 Strategy"):
        strategy_output = gr.Markdown(get_strategy())
        gr.Button("🔄 Analyze").click(fn=get_strategy, outputs=strategy_output)
    
    with gr.Tab("⚡ Manual Trigger"):
        gr.Markdown("Manually trigger an agent cycle:")
        trigger_output = gr.Markdown("")
        gr.Button("🚀 Run Cycle Now").click(fn=run_manual_cycle, outputs=trigger_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
