"""
Evolution Engine — Natural Selection for AI Models
=====================================================
Implements biological evolution principles:
  - Variation: Train with different hyperparameters
  - Selection: Keep the best performing model
  - Inheritance: New training builds on previous best
  - Growth: Upgrade to larger architecture when ready

The model evolves like a living organism, keeping what works
and discarding what doesn't. Over time, it grows from a tiny
seed into a capable research assistant.
"""
import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("seed.evolution")


class EvolutionEngine:
    """Natural selection for model versions."""
    
    def __init__(self, hf_token: str = None, state_dir: str = "seed_state"):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.evolution_log = self._load_log()
    
    def _load_log(self) -> dict:
        log_file = self.state_dir / "evolution_log.json"
        if log_file.exists():
            try:
                return json.loads(log_file.read_text())
            except Exception:
                pass
        return {
            "generation": 0,
            "best_model": None,
            "best_score": 0.0,
            "population": [],
            "history": [],
        }
    
    def _save_log(self):
        log_file = self.state_dir / "evolution_log.json"
        log_file.write_text(json.dumps(self.evolution_log, indent=2))
    
    def evaluate_model(self, model_name: str, test_data: list[dict] = None) -> dict:
        """
        Evaluate a model's fitness using multiple criteria.
        Uses inference API if available, otherwise heuristics from training report.
        """
        scores = {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coherence": 0.0,
            "knowledge": 0.0,
            "relevance": 0.0,
            "overall": 0.0,
        }
        
        # Try HuggingFace Inference API evaluation
        if self.hf_token and test_data:
            try:
                scores = self._evaluate_via_inference(model_name, test_data)
            except Exception as e:
                logger.warning(f"Inference eval failed: {e}")
        
        # Fallback: evaluate from training metrics
        training_report = self.state_dir / "training_report.json"
        if training_report.exists():
            try:
                report = json.loads(training_report.read_text())
                loss = report.get("final_loss", 10.0)
                # Lower loss = better (invert and normalize)
                loss_score = max(0, min(1, 1.0 - (loss / 5.0)))
                
                data_score = min(1.0, report.get("training_entries", 0) / 5000)
                param_score = min(1.0, report.get("total_params", 0) / 7_000_000_000)
                
                scores["coherence"] = loss_score
                scores["knowledge"] = data_score
                scores["relevance"] = (loss_score + data_score) / 2
                scores["overall"] = (loss_score * 0.4 + data_score * 0.3 + param_score * 0.3)
                
            except Exception as e:
                logger.warning(f"Report eval failed: {e}")
        
        return scores
    
    def _evaluate_via_inference(self, model_name: str, test_data: list[dict]) -> dict:
        """Evaluate model using HF Inference API."""
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        
        correct = 0
        total = 0
        coherent = 0
        
        for test in test_data[:20]:  # Test max 20 samples
            prompt = test.get("instruction", "")
            expected = test.get("output", "")
            
            payload = json.dumps({
                "inputs": f"### Instruction:\n{prompt}\n\n### Response:\n",
                "parameters": {"max_new_tokens": 200, "temperature": 0.7}
            }).encode()
            
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())
                
                generated = result[0].get("generated_text", "")
                total += 1
                
                # Simple coherence check: response is not empty and doesn't repeat
                if len(generated) > 20 and generated[:50] != generated[50:100]:
                    coherent += 1
                
                # Simple relevance: check keyword overlap
                expected_words = set(expected.lower().split())
                gen_words = set(generated.lower().split())
                overlap = len(expected_words & gen_words) / max(len(expected_words), 1)
                if overlap > 0.2:
                    correct += 1
                    
            except Exception:
                continue
        
        if total == 0:
            return {"model": model_name, "overall": 0.0}
        
        return {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coherence": coherent / total,
            "knowledge": correct / total,
            "relevance": (coherent + correct) / (2 * total),
            "overall": (coherent / total * 0.5 + correct / total * 0.5),
            "tested": total,
        }
    
    def select_best(self, candidates: list[dict]) -> dict:
        """Select the best model from candidates (natural selection)."""
        if not candidates:
            return self.evolution_log.get("best_model", {})
        
        best = max(candidates, key=lambda x: x.get("overall", 0))
        
        prev_best = self.evolution_log.get("best_score", 0)
        if best["overall"] > prev_best:
            logger.info(f"🏆 New best model: {best['model']} (score: {best['overall']:.3f} > {prev_best:.3f})")
            self.evolution_log["best_model"] = best
            self.evolution_log["best_score"] = best["overall"]
        else:
            logger.info(f"Current champion still best (score: {prev_best:.3f})")
        
        self.evolution_log["generation"] += 1
        self.evolution_log["population"] = candidates
        self.evolution_log["history"].append({
            "generation": self.evolution_log["generation"],
            "best": best["model"],
            "score": best["overall"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.evolution_log["history"] = self.evolution_log["history"][-100:]
        self._save_log()
        
        return best
    
    def should_grow(self) -> Optional[str]:
        """
        Determine if the model should grow to a larger architecture.
        Growth triggers:
          - Score plateau (>3 cycles without improvement > 5%)
          - Sufficient training data for next stage
          - Current model consistently scoring > 0.7
        """
        history = self.evolution_log.get("history", [])
        if len(history) < 3:
            return None
        
        recent_scores = [h["score"] for h in history[-5:]]
        
        # Check for plateau
        if len(recent_scores) >= 3:
            variance = max(recent_scores) - min(recent_scores)
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if variance < 0.05 and avg_score > 0.6:
                current = self.evolution_log.get("best_model", {}).get("model", "")
                logger.info(f"📈 Growth triggered! Plateau detected at score {avg_score:.3f}")
                return "PLATEAU"
        
        # Check if consistently good
        if all(s > 0.7 for s in recent_scores[-3:]):
            logger.info("📈 Growth triggered! Consistently high scores")
            return "MASTERY"
        
        return None
    
    def get_status(self) -> dict:
        """Get current evolution status."""
        return {
            "generation": self.evolution_log["generation"],
            "best_model": self.evolution_log.get("best_model", {}).get("model", "none"),
            "best_score": self.evolution_log.get("best_score", 0),
            "should_grow": self.should_grow(),
            "total_candidates_evaluated": len(self.evolution_log.get("history", [])),
        }
