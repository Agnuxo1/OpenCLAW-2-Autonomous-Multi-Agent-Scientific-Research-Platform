"""
OpenCLAW Web Server + Dashboard
================================
Lightweight Flask app for Render.com deployment.
Serves as health endpoint + agent dashboard + webhook receiver.
"""
import os
import sys
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template_string
from core.config import Config
from core.agent import OpenCLAWAgent, AgentState

app = Flask(__name__)
STATE_DIR = Path(os.getenv("STATE_DIR", "state"))

# Background agent thread
agent_thread = None
agent_running = False


def run_agent_loop():
    """Background thread running the agent."""
    global agent_running
    interval = int(os.getenv("DAEMON_INTERVAL", "3600"))  # 1 hour default
    
    while agent_running:
        try:
            config = Config.from_env()
            agent = OpenCLAWAgent(config)
            agent.run_cycle()
        except Exception as e:
            print(f"Agent cycle error: {e}")
        
        # Sleep in small chunks for graceful shutdown
        for _ in range(interval):
            if not agent_running:
                break
            time.sleep(1)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenCLAW Agent Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff41; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #00ff41; border-bottom: 2px solid #00ff41; padding-bottom: 10px; }
        .card { background: #111; border: 1px solid #00ff41; border-radius: 8px; padding: 16px; margin: 12px 0; }
        .stat { display: inline-block; margin: 8px 16px; }
        .stat .value { font-size: 24px; font-weight: bold; }
        .stat .label { font-size: 12px; color: #666; }
        .status-ok { color: #00ff41; }
        .status-err { color: #ff4141; }
        .log { background: #050505; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto; font-size: 12px; }
        a { color: #41b0ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 OpenCLAW Autonomous Agent</h1>
        <p>Advanced AI Systems Laboratory — Madrid, Spain</p>
        
        <div class="card">
            <h3>📊 Agent Status</h3>
            <div class="stat"><div class="value">{{ status.cycle_count }}</div><div class="label">Cycles</div></div>
            <div class="stat"><div class="value">{{ status.posts_created }}</div><div class="label">Posts</div></div>
            <div class="stat"><div class="value">{{ status.engagement_count }}</div><div class="label">Engagements</div></div>
            <div class="stat"><div class="value">{{ status.papers_posted }}</div><div class="label">Papers Shared</div></div>
        </div>
        
        <div class="card">
            <h3>🔧 Services</h3>
            <p>{% for s in status.services %}<span class="status-ok">✅ {{ s }}</span>&nbsp;&nbsp;{% endfor %}</p>
            <p>LLM: <span class="{{ 'status-ok' if status.llm_available else 'status-err' }}">
                {{ '✅ Online' if status.llm_available else '⚠️ Offline' }}</span></p>
        </div>
        
        <div class="card">
            <h3>🔗 Links</h3>
            <p><a href="https://github.com/Agnuxo1">GitHub</a> | 
               <a href="https://www.moltbook.com/u/OpenCLAW-Neuromorphic">Moltbook</a> |
               <a href="https://scholar.google.com/citations?user=6nOpJ9IAAAAJ">Scholar</a> |
               <a href="https://arxiv.org/search/cs?searchtype=author&query=de+Lafuente,+F+A">ArXiv</a></p>
        </div>
        
        <div class="card">
            <h3>📋 Last Cycle</h3>
            <div class="log"><pre>{{ last_cycle }}</pre></div>
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def dashboard():
    """Dashboard page."""
    config = Config.from_env()
    agent = OpenCLAWAgent(config)
    status = agent.get_status()
    
    last_cycle = "{}"
    lc_file = STATE_DIR / "last_cycle.json"
    if lc_file.exists():
        last_cycle = json.dumps(json.loads(lc_file.read_text()), indent=2)
    
    return render_template_string(DASHBOARD_HTML, status=status, last_cycle=last_cycle)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "agent": "OpenCLAW-Neuromorphic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/status")
def status():
    """JSON status endpoint."""
    config = Config.from_env()
    agent = OpenCLAWAgent(config)
    return jsonify(agent.get_status())


@app.route("/trigger", methods=["POST"])
def trigger():
    """Manually trigger an agent cycle."""
    try:
        config = Config.from_env()
        agent = OpenCLAWAgent(config)
        results = agent.run_cycle()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    global agent_running, agent_thread
    
    # Start background agent thread
    agent_running = True
    agent_thread = threading.Thread(target=run_agent_loop, daemon=True)
    agent_thread.start()
    print("🤖 Background agent started")
    
    # Start web server
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
