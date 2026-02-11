"""
OpenCLAW Agent + SEED — HuggingFace Spaces Dashboard
======================================================
Gradio interface with background agent loop and autonomous model growth.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.agent import OpenCLAWAgent, AgentState
from core.strategy import StrategyReflector
from research.arxiv_fetcher import ArxivFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("openclaw")

STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
SEED_STATE_DIR = Path(os.getenv("SEED_STATE_DIR", "seed_state"))
SEED_DATA_DIR = Path(os.getenv("SEED_DATA_DIR", "seed_data"))

agent_running = False
cycle_log = []
seed_log = []


# ==========================================================================
# BACKGROUND AGENT + SEED GROWTH
# ==========================================================================
def run_background_agent():
    """Background thread: agent + SEED growth combined."""
    global agent_running, cycle_log, seed_log
    agent_running = True
    interval = int(os.getenv("DAEMON_INTERVAL", "3600"))
    
    agent = OpenCLAWAgent(state_dir=str(STATE_DIR))
    
    # Initialize SEED
    seed_engine = None
    try:
        from seed.growth_engine import GrowthEngine
        seed_engine = GrowthEngine(
            hf_token=os.environ.get("HF_TOKEN", ""),
            state_dir=str(SEED_STATE_DIR),
            data_dir=str(SEED_DATA_DIR),
        )
        logger.info("🌱 SEED Growth Engine initialized")
    except Exception as e:
        logger.warning(f"SEED init failed (will retry): {e}")
    
    cycle_num = 0
    while True:
        cycle_num += 1
        now = datetime.now(timezone.utc).isoformat()
        
        # === AGENT CYCLE ===
        try:
            results = agent.run_cycle()
            entry = f"[{now}] Agent cycle #{cycle_num}: " + ", ".join(
                f"{a['task']}={a['status']}" for a in results.get("actions", [])
            )
            cycle_log.append(entry)
            logger.info(entry)
        except Exception as e:
            cycle_log.append(f"[{now}] Agent error: {e}")
            logger.error(f"Agent cycle error: {e}")
        
        # === SEED GROWTH (every 6th cycle = ~6 hours) ===
        if seed_engine and cycle_num % 6 == 0:
            try:
                logger.info("🌱 Running SEED growth cycle...")
                seed_results = seed_engine.run_cycle()
                seed_entry = (
                    f"[{now}] SEED cycle #{seed_results.get('cycle', '?')}: "
                    f"stage={seed_engine.cycle_log.get('current_stage', '?')}, "
                    f"data={seed_engine.cycle_log.get('total_data_harvested', 0)}"
                )
                seed_log.append(seed_entry)
                logger.info(seed_entry)
            except Exception as e:
                seed_log.append(f"[{now}] SEED error: {e}")
                logger.error(f"SEED cycle error: {e}")
        elif seed_engine is None:
            # Retry SEED init
            try:
                from seed.growth_engine import GrowthEngine
                seed_engine = GrowthEngine(
                    hf_token=os.environ.get("HF_TOKEN", ""),
                    state_dir=str(SEED_STATE_DIR),
                    data_dir=str(SEED_DATA_DIR),
                )
            except Exception:
                pass
        
        # Keep logs bounded
        cycle_log[:] = cycle_log[-200:]
        seed_log[:] = seed_log[-100:]
        
        time.sleep(interval)


# Start background
bg_thread = threading.Thread(target=run_background_agent, daemon=True)
bg_thread.start()
logger.info("🚀 Background agent + SEED started")


# ==========================================================================
# DASHBOARD FUNCTIONS
# ==========================================================================
def get_status():
    state_file = STATE_DIR / "agent_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            return json.dumps(state, indent=2)
        except Exception:
            pass
    return json.dumps({
        "status": "running" if agent_running else "starting",
        "message": "Agent initializing...",
    }, indent=2)

def get_activity():
    return "\n".join(cycle_log[-50:]) if cycle_log else "No activity yet — first cycle in progress..."

def get_papers():
    fetcher = ArxivFetcher()
    papers = fetcher.known_papers
    lines = []
    for p in papers:
        lines.append(f"📄 {p['title']}")
        lines.append(f"   ID: {p['arxiv_id']} | Year: {p.get('year', '?')}")
        lines.append(f"   {p.get('abstract', '')[:150]}...")
        lines.append("")
    return "\n".join(lines) if lines else "No papers loaded yet."

def get_strategy():
    reflector = StrategyReflector(state_dir=str(STATE_DIR))
    report = reflector.analyze()
    return json.dumps(report, indent=2)

def get_seed_status():
    """Get SEED growth status."""
    try:
        from seed.growth_engine import GrowthEngine
        engine = GrowthEngine(
            hf_token=os.environ.get("HF_TOKEN", ""),
            state_dir=str(SEED_STATE_DIR),
            data_dir=str(SEED_DATA_DIR),
        )
        status = engine.get_status()
        return json.dumps(status, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "initializing", "note": str(e)}, indent=2)

def get_seed_log():
    return "\n".join(seed_log[-50:]) if seed_log else "SEED not yet active — first growth cycle pending..."

def trigger_harvest():
    """Manually trigger data harvest."""
    try:
        from seed.data.harvester import DataHarvester
        h = DataHarvester(str(SEED_DATA_DIR))
        stats = h.harvest_all()
        return f"✅ Harvested {stats['total']} entries:\n" + json.dumps(stats, indent=2)
    except Exception as e:
        return f"❌ Harvest failed: {e}"

def trigger_cycle():
    try:
        agent = OpenCLAWAgent(state_dir=str(STATE_DIR))
        results = agent.run_cycle()
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return f"Error: {e}"


# ==========================================================================
# GRADIO INTERFACE
# ==========================================================================
with gr.Blocks(title="🌱 OpenCLAW SEED — Self-Evolving Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌱 OpenCLAW SEED — Self-Evolving Autonomous Agent
    *A seed that grows into a tree. Autonomous 24/7 research agent with self-training AI.*
    
    **By Francisco Angulo de Lafuente** | [GitHub](https://github.com/Agnuxo1) | [Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ)
    """)
    
    with gr.Tab("📊 Agent Status"):
        status_box = gr.Code(label="Agent State", language="json")
        gr.Button("Refresh").click(get_status, outputs=status_box)
        demo.load(get_status, outputs=status_box)
    
    with gr.Tab("📝 Activity Log"):
        log_box = gr.Textbox(label="Recent Activity", lines=20, max_lines=30)
        gr.Button("Refresh").click(get_activity, outputs=log_box)
        demo.load(get_activity, outputs=log_box)
    
    with gr.Tab("🌱 SEED Growth"):
        gr.Markdown("""
        ### Self-Evolving Model Growth
        SEED harvests knowledge, trains itself, and grows autonomously.
        
        **Growth stages:** GERMINATION (0.5B) → SEEDLING (1B) → SAPLING (3B) → YOUNG_TREE (7B) → MATURE_TREE (13B+)
        """)
        seed_status_box = gr.Code(label="SEED Status", language="json")
        seed_log_box = gr.Textbox(label="Growth Log", lines=10)
        with gr.Row():
            gr.Button("🔄 Refresh Status").click(get_seed_status, outputs=seed_status_box)
            gr.Button("🌾 Harvest Data Now").click(trigger_harvest, outputs=seed_log_box)
        demo.load(get_seed_status, outputs=seed_status_box)
    
    with gr.Tab("📄 Research Papers"):
        papers_box = gr.Textbox(label="Known Papers", lines=20, max_lines=30)
        gr.Button("Refresh").click(get_papers, outputs=papers_box)
        demo.load(get_papers, outputs=papers_box)
    
    with gr.Tab("🧠 Strategy"):
        strategy_box = gr.Code(label="Strategy Analysis", language="json")
        gr.Button("Analyze").click(get_strategy, outputs=strategy_box)
    
    with gr.Tab("⚡ Manual Trigger"):
        trigger_box = gr.Code(label="Cycle Results", language="json")
        gr.Button("🤖 Run Agent Cycle").click(trigger_cycle, outputs=trigger_box)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
