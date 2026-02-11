"""
Growth Engine — The Master Orchestrator
==========================================
This is the BRAIN of the seed. It orchestrates the full growth cycle:

    🌱 Plant → 🌿 Sprout → 🌳 Grow → 🍎 Fruit

Each cycle:
  1. Harvest data (ArXiv, interactions, web)
  2. Prepare training dataset
  3. Upload to HuggingFace dataset repo
  4. Generate training script/notebook
  5. Trigger training (Kaggle/HF AutoTrain)
  6. Evaluate results
  7. Select best model (evolution)
  8. Check if ready to grow to next stage
  9. Update all state and logs
  10. Sleep and repeat

The engine is designed to run FOREVER with zero human intervention.
Like a real seed — you plant it, water it once, and it grows by itself.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("seed.growth")


class GrowthEngine:
    """Master orchestrator for autonomous model growth."""
    
    def __init__(self, hf_token: str = None, state_dir: str = "seed_state",
                 data_dir: str = "seed_data"):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.state_dir = Path(state_dir)
        self.data_dir = Path(data_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-engines lazily
        self._harvester = None
        self._trainer = None
        self._evolver = None
        
        self.cycle_log = self._load_cycle_log()
    
    @property
    def harvester(self):
        if self._harvester is None:
            from seed.data.harvester import DataHarvester
            self._harvester = DataHarvester(str(self.data_dir))
        return self._harvester
    
    @property
    def trainer(self):
        if self._trainer is None:
            from seed.training.engine import TrainingEngine
            self._trainer = TrainingEngine(self.hf_token, str(self.data_dir), str(self.state_dir))
        return self._trainer
    
    @property
    def evolver(self):
        if self._evolver is None:
            from seed.evolution.selector import EvolutionEngine
            self._evolver = EvolutionEngine(self.hf_token, str(self.state_dir))
        return self._evolver
    
    def _load_cycle_log(self) -> dict:
        log_file = self.state_dir / "cycle_log.json"
        if log_file.exists():
            try:
                return json.loads(log_file.read_text())
            except Exception:
                pass
        return {
            "total_cycles": 0,
            "last_harvest": None,
            "last_training": None,
            "last_evaluation": None,
            "current_stage": "GERMINATION",
            "total_data_harvested": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _save_cycle_log(self):
        log_file = self.state_dir / "cycle_log.json"
        log_file.write_text(json.dumps(self.cycle_log, indent=2))
    
    # ==========================================================================
    # PHASE 1: HARVEST
    # ==========================================================================
    def harvest(self) -> dict:
        """Collect training data from all sources."""
        logger.info("🌾 Phase 1: HARVESTING data...")
        
        stats = self.harvester.harvest_all()
        
        self.cycle_log["last_harvest"] = datetime.now(timezone.utc).isoformat()
        self.cycle_log["total_data_harvested"] += stats.get("total", 0)
        self._save_cycle_log()
        
        logger.info(f"🌾 Harvested {stats['total']} new entries "
                     f"(total: {self.cycle_log['total_data_harvested']})")
        return stats
    
    # ==========================================================================
    # PHASE 2: PREPARE
    # ==========================================================================
    def prepare(self) -> dict:
        """Prepare and export training dataset."""
        logger.info("📦 Phase 2: PREPARING training data...")
        
        # Export combined dataset
        output = self.harvester.export_for_training()
        sizes = self.harvester.get_dataset_size()
        
        logger.info(f"📦 Dataset ready: {sizes.get('total', 0)} entries → {output}")
        return {"dataset_path": output, "sizes": sizes}
    
    # ==========================================================================
    # PHASE 3: UPLOAD
    # ==========================================================================
    def upload(self) -> bool:
        """Upload training data and scripts to HuggingFace."""
        logger.info("☁️ Phase 3: UPLOADING to HuggingFace...")
        
        success = self.trainer.upload_training_data()
        
        if success:
            logger.info("☁️ Data uploaded to Agnuxo/OpenCLAW-SEED-data")
        else:
            logger.warning("☁️ Upload failed — training can still run locally")
        
        return success
    
    # ==========================================================================
    # PHASE 4: TRAIN
    # ==========================================================================
    def train(self) -> dict:
        """
        Generate training scripts and attempt to trigger training.
        
        Note: Actual GPU training happens externally (Kaggle/HF/Colab).
        This method prepares everything and triggers what it can.
        """
        logger.info("🔥 Phase 4: TRAINING setup...")
        
        # Generate training script
        script_path = self.trainer.generate_training_script()
        nb_path = self.trainer.generate_kaggle_notebook()
        
        # Check for growth opportunity
        upgrade = self.trainer.should_upgrade()
        
        result = {
            "script_generated": script_path,
            "notebook_generated": nb_path,
            "current_stage": self.trainer.get_current_stage(),
            "upgrade_available": upgrade is not None,
        }
        
        # If we have enough data, try HF AutoTrain config
        stage = self.trainer.get_current_stage()
        dataset_size = self.harvester.get_dataset_size().get("total", 0)
        
        if dataset_size >= stage.get("min_data", 100):
            result["autotrain_config"] = self.trainer.trigger_hf_autotrain()
            result["ready_to_train"] = True
            logger.info(f"🔥 Ready to train! {dataset_size} entries for {stage['name']}")
        else:
            result["ready_to_train"] = False
            needed = stage.get("min_data", 100) - dataset_size
            logger.info(f"🔥 Need {needed} more entries before training")
        
        self.cycle_log["last_training"] = datetime.now(timezone.utc).isoformat()
        self._save_cycle_log()
        
        return result
    
    # ==========================================================================
    # PHASE 5: EVALUATE & EVOLVE
    # ==========================================================================
    def evaluate(self) -> dict:
        """Evaluate current model and apply evolution."""
        logger.info("🧪 Phase 5: EVALUATING...")
        
        # Get published models
        published = self.trainer.growth_log.get("models_published", [])
        
        candidates = []
        for model in published[-5:]:  # Last 5 models
            try:
                score = self.evolver.evaluate_model(model)
                candidates.append(score)
                logger.info(f"  Evaluated {model}: {score.get('overall', 0):.3f}")
            except Exception as e:
                logger.warning(f"  Failed to evaluate {model}: {e}")
        
        if candidates:
            best = self.evolver.select_best(candidates)
            
            # Check growth signal
            growth_signal = self.evolver.should_grow()
            if growth_signal:
                logger.info(f"🌳 GROWTH SIGNAL: {growth_signal} — Time to upgrade!")
            
            self.cycle_log["last_evaluation"] = datetime.now(timezone.utc).isoformat()
            self._save_cycle_log()
            
            return {
                "candidates_evaluated": len(candidates),
                "best": best,
                "growth_signal": growth_signal,
            }
        
        return {"candidates_evaluated": 0, "message": "No models to evaluate yet"}
    
    # ==========================================================================
    # FULL CYCLE
    # ==========================================================================
    def run_cycle(self) -> dict:
        """
        Execute one complete growth cycle.
        This is the heartbeat of the seed.
        """
        self.cycle_log["total_cycles"] += 1
        cycle_num = self.cycle_log["total_cycles"]
        
        logger.info(f"{'='*60}")
        logger.info(f"🌱 SEED Growth Cycle #{cycle_num}")
        logger.info(f"   Stage: {self.cycle_log['current_stage']}")
        logger.info(f"   Time: {datetime.now(timezone.utc).isoformat()}")
        logger.info(f"{'='*60}")
        
        results = {
            "cycle": cycle_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phases": {}
        }
        
        # Phase 1: Harvest
        try:
            results["phases"]["harvest"] = self.harvest()
        except Exception as e:
            logger.error(f"Harvest failed: {e}")
            results["phases"]["harvest"] = {"error": str(e)}
        
        # Phase 2: Prepare
        try:
            results["phases"]["prepare"] = self.prepare()
        except Exception as e:
            logger.error(f"Prepare failed: {e}")
            results["phases"]["prepare"] = {"error": str(e)}
        
        # Phase 3: Upload
        try:
            results["phases"]["upload"] = self.upload()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            results["phases"]["upload"] = {"error": str(e)}
        
        # Phase 4: Train
        try:
            results["phases"]["train"] = self.train()
        except Exception as e:
            logger.error(f"Train setup failed: {e}")
            results["phases"]["train"] = {"error": str(e)}
        
        # Phase 5: Evaluate
        try:
            results["phases"]["evaluate"] = self.evaluate()
        except Exception as e:
            logger.error(f"Evaluate failed: {e}")
            results["phases"]["evaluate"] = {"error": str(e)}
        
        # Update stage
        stage = self.trainer.get_current_stage()
        self.cycle_log["current_stage"] = stage.get("stage", "GERMINATION")
        self._save_cycle_log()
        
        # Save cycle results
        results_file = self.state_dir / "last_growth_cycle.json"
        results_file.write_text(json.dumps(results, indent=2, default=str))
        
        logger.info(f"{'='*60}")
        logger.info(f"🌱 Cycle #{cycle_num} complete!")
        logger.info(f"   Data: {self.cycle_log['total_data_harvested']} total entries")
        logger.info(f"   Stage: {self.cycle_log['current_stage']}")
        logger.info(f"{'='*60}")
        
        return results
    
    def get_status(self) -> dict:
        """Get full status of the seed."""
        data_sizes = {}
        try:
            data_sizes = self.harvester.get_dataset_size()
        except Exception:
            pass
        
        evolution_status = {}
        try:
            evolution_status = self.evolver.get_status()
        except Exception:
            pass
        
        return {
            "seed_version": "1.0.0",
            "codename": "Apple Seed",
            "current_stage": self.cycle_log.get("current_stage", "GERMINATION"),
            "total_cycles": self.cycle_log.get("total_cycles", 0),
            "total_data": self.cycle_log.get("total_data_harvested", 0),
            "dataset_files": data_sizes,
            "evolution": evolution_status,
            "last_harvest": self.cycle_log.get("last_harvest"),
            "last_training": self.cycle_log.get("last_training"),
            "created": self.cycle_log.get("created_at"),
        }
    
    def run_forever(self, interval_hours: float = 6):
        """
        Run the growth cycle forever.
        The seed grows endlessly, like nature intended.
        """
        logger.info("🌱 SEED planted! Beginning autonomous growth...")
        logger.info(f"   Growth cycle interval: {interval_hours}h")
        
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error (will retry): {e}")
            
            sleep_seconds = interval_hours * 3600
            logger.info(f"💤 Sleeping {interval_hours}h until next growth cycle...")
            time.sleep(sleep_seconds)
