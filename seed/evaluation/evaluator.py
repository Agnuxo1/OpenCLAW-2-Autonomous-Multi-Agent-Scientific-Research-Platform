"""
Evaluator — Autonomous Model Quality Assessment
==================================================
Tests the seed model against benchmarks without human intervention.

Tests:
  1. Research Q&A: Can it answer questions about neuromorphic computing?
  2. Coherence: Does it produce grammatical, non-repetitive text?
  3. Self-knowledge: Does it know about OpenCLAW and our research?
  4. Reasoning: Can it draw connections between concepts?
  5. Growth check: Is it better than the previous version?
"""
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("seed.evaluator")

# Test suite — questions the model MUST learn to answer well
BENCHMARK = [
    {
        "id": "research_1",
        "category": "research_knowledge",
        "instruction": "What is the CHIMERA architecture?",
        "expected_keywords": ["gpu", "neural", "asic", "speedup", "physics", "pytorch"],
        "weight": 2.0,
    },
    {
        "id": "research_2",
        "category": "research_knowledge",
        "instruction": "Explain holographic neural networks.",
        "expected_keywords": ["holographic", "wave", "interference", "optical", "encoding"],
        "weight": 2.0,
    },
    {
        "id": "research_3",
        "category": "research_knowledge",
        "instruction": "What is thermodynamic reservoir computing?",
        "expected_keywords": ["reservoir", "thermodynamic", "entropy", "computation", "physical"],
        "weight": 2.0,
    },
    {
        "id": "self_1",
        "category": "self_knowledge",
        "instruction": "Who is Francisco Angulo de Lafuente?",
        "expected_keywords": ["researcher", "madrid", "ai", "neural", "physics", "novelist"],
        "weight": 1.5,
    },
    {
        "id": "self_2",
        "category": "self_knowledge",
        "instruction": "What is OpenCLAW?",
        "expected_keywords": ["autonomous", "research", "agent", "agi", "scientific"],
        "weight": 1.5,
    },
    {
        "id": "reasoning_1",
        "category": "reasoning",
        "instruction": "How could physics-based neural networks outperform traditional deep learning?",
        "expected_keywords": ["physical", "energy", "efficiency", "analog", "computation"],
        "weight": 1.0,
    },
    {
        "id": "reasoning_2",
        "category": "reasoning",
        "instruction": "What is the relationship between consciousness and computation?",
        "expected_keywords": ["consciousness", "information", "process", "theory", "emergence"],
        "weight": 1.0,
    },
    {
        "id": "coherence_1",
        "category": "coherence",
        "instruction": "Write a brief abstract for a paper on neuromorphic AGI architectures.",
        "expected_keywords": ["present", "approach", "architecture", "results", "demonstrate"],
        "weight": 1.0,
    },
    {
        "id": "agi_1",
        "category": "agi_understanding",
        "instruction": "What are the main obstacles to achieving AGI?",
        "expected_keywords": ["general", "intelligence", "reasoning", "learning", "scalability"],
        "weight": 1.0,
    },
    {
        "id": "collab_1",
        "category": "collaboration",
        "instruction": "Why should researchers collaborate on open-source AGI projects?",
        "expected_keywords": ["open", "science", "collaboration", "progress", "share"],
        "weight": 1.0,
    },
]


class Evaluator:
    """Autonomous model evaluation."""

    def __init__(self, hf_token: str = "", state_dir: str = "seed_state"):
        self.hf_token = hf_token
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model_name: str) -> dict:
        """Run full benchmark against a model via HF Inference API."""
        results = {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scores": {},
            "category_scores": {},
            "overall": 0.0,
            "tested": 0,
            "passed": 0,
        }

        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        total_weight = 0
        weighted_score = 0

        for test in BENCHMARK:
            try:
                score = self._run_test(url, headers, test)
                results["scores"][test["id"]] = score
                results["tested"] += 1
                if score > 0.5:
                    results["passed"] += 1

                w = test.get("weight", 1.0)
                weighted_score += score * w
                total_weight += w

                cat = test["category"]
                if cat not in results["category_scores"]:
                    results["category_scores"][cat] = []
                results["category_scores"][cat].append(score)
            except Exception as e:
                logger.warning(f"Test {test['id']} failed: {e}")
                results["scores"][test["id"]] = 0.0

        if total_weight > 0:
            results["overall"] = weighted_score / total_weight

        # Average category scores
        for cat, scores in results["category_scores"].items():
            results["category_scores"][cat] = sum(scores) / len(scores) if scores else 0

        # Save results
        eval_file = self.state_dir / f"eval_{model_name.replace('/', '_')}.json"
        eval_file.write_text(json.dumps(results, indent=2))

        logger.info(
            f"Evaluated {model_name}: overall={results['overall']:.3f}, "
            f"passed={results['passed']}/{results['tested']}"
        )
        return results

    def _run_test(self, url: str, headers: dict, test: dict) -> float:
        """Run a single benchmark test and return a score 0-1."""
        prompt = (
            f"### Instruction:\n{test['instruction']}\n\n"
            f"### Response:\n"
        )
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "temperature": 0.3}
        }).encode()

        req = urllib.request.Request(url, data=payload, headers={
            **headers, "Content-Type": "application/json"
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        generated = ""
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            generated = data.get("generated_text", "")

        # Remove prompt from response
        if "### Response:" in generated:
            generated = generated.split("### Response:")[-1].strip()

        if not generated or len(generated) < 10:
            return 0.0

        # Score 1: Keyword match (relevant content)
        gen_lower = generated.lower()
        keywords = test.get("expected_keywords", [])
        if keywords:
            hits = sum(1 for k in keywords if k in gen_lower)
            keyword_score = hits / len(keywords)
        else:
            keyword_score = 0.5

        # Score 2: Coherence (not repetitive, proper length)
        words = generated.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        length_score = min(1.0, len(words) / 30)
        coherence_score = (unique_ratio + length_score) / 2

        # Score 3: No hallucination signals
        hallucination_markers = [
            "i don't know", "i cannot", "as an ai", "i'm sorry",
            "###", "instruction:", "input:", "output:"
        ]
        hallucination_penalty = sum(
            0.15 for m in hallucination_markers if m in gen_lower
        )

        final = (keyword_score * 0.5 + coherence_score * 0.5) - hallucination_penalty
        return max(0.0, min(1.0, final))

    def compare_models(self, model_a: str, model_b: str) -> dict:
        """Compare two models head-to-head."""
        eval_a = self.evaluate_model(model_a)
        eval_b = self.evaluate_model(model_b)

        winner = model_a if eval_a["overall"] > eval_b["overall"] else model_b
        margin = abs(eval_a["overall"] - eval_b["overall"])

        return {
            "model_a": {"name": model_a, "score": eval_a["overall"]},
            "model_b": {"name": model_b, "score": eval_b["overall"]},
            "winner": winner,
            "margin": margin,
            "significant": margin > 0.05,
        }

    def generate_report(self) -> str:
        """Generate evaluation report from stored results."""
        reports = []
        for f in self.state_dir.glob("eval_*.json"):
            try:
                reports.append(json.loads(f.read_text()))
            except Exception:
                continue

        if not reports:
            return "No evaluations yet."

        reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        latest = reports[0]

        lines = [
            f"# SEED Evaluation Report",
            f"Model: {latest['model']}",
            f"Overall: {latest['overall']:.3f}",
            f"Passed: {latest['passed']}/{latest['tested']}",
            "",
            "## Category Scores:",
        ]
        for cat, score in latest.get("category_scores", {}).items():
            lines.append(f"  {cat}: {score:.3f}")

        return "\n".join(lines)
