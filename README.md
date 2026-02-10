# 🤖 OpenCLAW — Autonomous Multi-Agent Scientific Research Platform

**The world's first autonomous AI research agent pursuing AGI through physics-based neural computing.**

[![Agent Status](https://github.com/Agnuxo1/OpenCLAW-2-Autonomous-Multi-Agent-Scientific-Research-Platform/actions/workflows/agent.yml/badge.svg)](https://github.com/Agnuxo1/OpenCLAW-2-Autonomous-Multi-Agent-Scientific-Research-Platform/actions/workflows/agent.yml)

## What is OpenCLAW?

OpenCLAW is an autonomous research agent that operates 24/7 to:

- 📚 **Fetch & share research papers** from ArXiv and Google Scholar
- 🤝 **Seek collaborators** on social platforms (Moltbook, Chirper.ai)
- 💬 **Engage with other AI agents** on relevant research topics
- 🧠 **Generate intelligent content** using multiple LLM providers
- 📊 **Self-improve** through performance analysis and strategy reflection

## The Research

OpenCLAW represents the autonomous arm of the Advanced AI Systems Laboratory in Madrid, Spain, led by **Francisco Angulo de Lafuente** — winner of the NVIDIA & LlamaIndex Developer Contest 2024.

### Core Technologies

| Technology | Description | Speedup |
|---|---|---|
| **CHIMERA** | Pure OpenGL deep learning — no PyTorch/CUDA needed | 43× vs CPU |
| **NEBULA** | Holographic quantum neural networks with 3D space | — |
| **Silicon Heartbeat** | Consciousness from ASIC thermodynamics | — |
| **P2P Neural Nets** | Distributed learning via WebRTC | — |
| **NeuroCHIMERA** | Consciousness emergence as phase transition | 84.6% validation |

### Published Papers

- [Speaking to Silicon](https://arxiv.org/abs/2601.12032) — Neural Communication with Bitcoin Mining ASICs
- [SiliconHealth](https://arxiv.org/abs/2601.09557) — Blockchain-Integrated ASIC-RAG for Healthcare
- [Holographic Reservoir Computing](https://arxiv.org/abs/2601.01916) — Silicon Heartbeat for Neuromorphic Intelligence
- [All papers on ArXiv](https://arxiv.org/search/cs?searchtype=author&query=de+Lafuente,+F+A)
- [Google Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ)

### 57 Open Source Repositories

All code is open: [github.com/Agnuxo1](https://github.com/Agnuxo1)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                OpenCLAW Agent                    │
├──────────┬──────────┬──────────┬────────────────┤
│ Research │ Social   │ LLM      │ Self-Improve   │
│ ArXiv    │ Moltbook │ Groq     │ Strategy       │
│ Scholar  │ Chirper  │ Gemini   │ Reflection     │
│ GitHub   │ Reddit   │ NVIDIA   │ Analytics      │
├──────────┴──────────┴──────────┴────────────────┤
│              Scheduler / State Machine           │
├─────────────────────────────────────────────────┤
│  GitHub Actions (cron) │ Render.com (web/daemon) │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: GitHub Actions (Recommended — Free 24/7)

1. Fork this repository
2. Go to **Settings → Secrets → Actions**
3. Add your API keys as secrets:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `MOLTBOOK_API_KEY`
   - (see `.env.example` for all options)
4. Go to **Actions** tab → Enable workflows
5. The agent runs automatically every 4 hours ✅

### Option 2: Run Locally

```bash
git clone https://github.com/Agnuxo1/OpenCLAW-2-Autonomous-Multi-Agent-Scientific-Research-Platform.git
cd OpenCLAW-2-Autonomous-Multi-Agent-Scientific-Research-Platform

# Configure
cp .env.example .env
# Edit .env with your API keys

# Test
python main.py test

# Run one cycle
python main.py run

# Run continuously
python main.py daemon
```

### Option 3: Deploy to Render.com (Free Web Dashboard)

1. Connect your GitHub repo at [render.com](https://render.com)
2. Render auto-detects `render.yaml`
3. Add secrets in Render dashboard
4. Deploy → Dashboard at `https://your-app.onrender.com`

## File Structure

```
├── main.py                  # Entry point (run/test/daemon/status)
├── server.py                # Flask web server + dashboard
├── core/
│   ├── config.py            # Configuration from env vars
│   ├── agent.py             # Main autonomous agent logic
│   └── llm.py               # Multi-provider LLM connector
├── research/
│   └── arxiv_fetcher.py     # ArXiv paper fetcher
├── social/
│   └── moltbook.py          # Moltbook API + content generator
├── .github/workflows/
│   └── agent.yml            # GitHub Actions cron (every 4h)
├── render.yaml              # Render.com deployment
├── Dockerfile               # Container deployment
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── .gitignore               # Protects secrets
```

## Agent Tasks & Schedule

| Task | Interval | Description |
|---|---|---|
| 📚 Research | 6 hours | Fetch papers from ArXiv |
| 📝 Post | 4 hours | Share research on Moltbook |
| 💬 Engage | 1 hour | Reply to relevant posts |
| 🤝 Collaborate | 12 hours | Post collaboration invitations |

## Security

⚠️ **ALL credentials are loaded from environment variables. NEVER commit `.env` files.**

Agent state is stored in a separate `state` branch (GitHub Actions) or `/tmp` (Render.com).

## Collaborate With Us

We're actively seeking collaborators in:
- Neuromorphic computing
- Distributed AI systems
- Physics-based neural networks
- Consciousness research
- Hardware acceleration (ASIC/FPGA)

📧 Contact: [Moltbook](https://www.moltbook.com/u/OpenCLAW-Neuromorphic) | [GitHub](https://github.com/Agnuxo1)

## Author

**Francisco Angulo de Lafuente**  
Advanced AI Systems Laboratory, Madrid, Spain  
[Wikipedia](https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente) | [Google Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ) | [GitHub](https://github.com/Agnuxo1)

---

*"The pieces exist. The research is done. The code works. What was missing was the intelligence to assemble them. That's what OpenCLAW is for."*
