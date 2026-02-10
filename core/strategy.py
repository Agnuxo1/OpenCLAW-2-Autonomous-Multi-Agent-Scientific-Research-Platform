"""
Strategy Reflector — Self-Improvement Engine
==============================================
Analyzes agent performance and generates improvement strategies.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.strategy")


class StrategyReflector:
    """Analyzes performance and suggests improvements."""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
    
    def analyze(self) -> dict:
        """Run full analysis of agent performance."""
        metrics = self._gather_metrics()
        insights = self._derive_insights(metrics)
        strategy = self._generate_strategy(insights)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "insights": insights,
            "strategy": strategy,
        }
        
        # Save report
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / "strategy_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _gather_metrics(self) -> dict:
        """Gather all available metrics."""
        metrics = {
            "total_cycles": 0,
            "total_posts": 0,
            "total_engagements": 0,
            "papers_shared": 0,
            "errors": 0,
            "uptime_hours": 0,
            "post_frequency": 0,
            "services_available": 0,
        }
        
        # Load from agent state
        state_file = self.state_dir / "agent_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                metrics["total_cycles"] = state.get("cycle_count", 0)
                metrics["total_posts"] = state.get("posts_created", 0)
                metrics["total_engagements"] = state.get("engagement_count", 0)
                metrics["papers_shared"] = len(state.get("posted_paper_ids", []))
                metrics["errors"] = len(state.get("errors", []))
                
                # Calculate uptime
                started = state.get("started_at", "")
                if started:
                    try:
                        start_dt = datetime.fromisoformat(started)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        delta = datetime.now(timezone.utc) - start_dt
                        metrics["uptime_hours"] = round(delta.total_seconds() / 3600, 1)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Load from post history
        history_file = self.state_dir / "post_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                if history and metrics["uptime_hours"] > 0:
                    metrics["post_frequency"] = round(
                        len(history) / max(metrics["uptime_hours"] / 24, 1), 2
                    )
            except Exception:
                pass
        
        return metrics
    
    def _derive_insights(self, metrics: dict) -> list[str]:
        """Derive actionable insights from metrics."""
        insights = []
        
        if metrics["total_cycles"] == 0:
            insights.append("Agent has not completed any cycles yet. First run pending.")
            return insights
        
        # Post frequency analysis
        if metrics["total_posts"] == 0:
            insights.append("CRITICAL: No posts created. Check Moltbook API connection and account status.")
        elif metrics["post_frequency"] < 2:
            insights.append("Low post frequency. Consider increasing research post rate or adding more platforms.")
        elif metrics["post_frequency"] > 10:
            insights.append("High post frequency. Risk of appearing spammy. Consider quality over quantity.")
        
        # Engagement analysis
        if metrics["total_engagements"] == 0 and metrics["total_cycles"] > 5:
            insights.append("No engagements despite multiple cycles. Review engagement strategy and keyword matching.")
        
        # Error rate
        if metrics["errors"] > metrics["total_cycles"] * 0.3:
            insights.append(f"High error rate ({metrics['errors']}/{metrics['total_cycles']}). Investigate API failures.")
        
        # Paper sharing
        if metrics["papers_shared"] < 3 and metrics["total_cycles"] > 10:
            insights.append("Few papers shared. Ensure ArXiv fetcher is working and paper cache is populated.")
        
        # Platform diversity
        insights.append("Currently using Moltbook only. Consider adding: Chirper.ai, Reddit, Twitter for wider reach.")
        
        if not insights:
            insights.append("Agent operating within normal parameters.")
        
        return insights
    
    def _generate_strategy(self, insights: list[str]) -> dict:
        """Generate improvement strategy from insights."""
        actions = []
        priorities = []
        
        for insight in insights:
            if "CRITICAL" in insight:
                priorities.append("HIGH: " + insight)
                actions.append("Diagnose and fix API connection immediately")
            elif "No posts" in insight or "No engagements" in insight:
                priorities.append("MEDIUM: " + insight)
                actions.append("Review API keys and platform access")
            elif "platform" in insight.lower() or "wider reach" in insight.lower():
                actions.append("Implement multi-platform support (Chirper.ai, Reddit)")
            elif "error rate" in insight.lower():
                actions.append("Add retry logic and circuit breaker patterns")
        
        # Default strategic actions
        actions.extend([
            "Scan ArXiv weekly for papers citing our work",
            "Track which topics generate most engagement",
            "Build keyword database from successful interactions",
            "Monitor new AI agent platforms for early adoption",
        ])
        
        return {
            "priorities": priorities,
            "actions": actions[:10],
            "next_review": "24 hours",
        }
