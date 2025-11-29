"""
Optuna Hyperparameter Tuner for ACO Graph Coloring
Provides flexible hyperparameter optimization with study persistence and resumption.
"""

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import json
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import warnings

# Suppress experimental warning for JournalStorage
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

from objective_function import _trial_graph_viz_data
from visualization_utils import (
    generate_trial_visualizations,
    plot_study_history,
    plot_param_importances,
    plot_slice,
    plot_timeline
)


class OptunaACOTuner:
    """
    Optuna-based hyperparameter tuner for ACO algorithms.
    
    Features:
    - JSON-based study persistence and resumption
    - Automatic tracking of remaining trials
    - Per-trial result saving for each graph
    - Comprehensive visualization of optimization progress
    """
    
    def __init__(self, study_name: str, data_root: str, direction: str = "minimize"):
        """
        Initialize the Optuna tuner.
        
        Args:
            study_name: Name of the study (used for saving/loading)
            data_root: Absolute path to the data directory
            direction: Optimization direction ('minimize' or 'maximize')
        """
        self.study_name = study_name
        self.direction = direction
        self.data_root = Path(data_root)
        
        # Setup study-specific folder structure
        self.study_folder = self.data_root / "studies" / study_name
        self.results_path = self.study_folder / "results"
        self.figures_path = self.study_folder / "figures"
        
        # Create directories
        self.study_folder.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # Setup JSON storage
        self.journal_file = self.study_folder / f"{study_name}.log"
        self.storage = JournalStorage(JournalFileStorage(str(self.journal_file)))
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        self.best_value: Optional[float] = None
    
    def create_or_load_study(self) -> optuna.Study:
        """Create a new study or load an existing one from JSON storage."""
        try:
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage)
            completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"✓ Loaded existing study '{self.study_name}'")
            print(f"  Total trials: {len(self.study.trials)} (Completed: {completed})")
            print(f"  Storage: {self.journal_file}")
        except KeyError:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                load_if_exists=False
            )
            print(f"✓ Created new study '{self.study_name}'")
            print(f"  Storage: {self.journal_file}")
        
        return self.study
    
    def _get_remaining_trials(self, total_trials: int) -> int:
        """Calculate remaining trials based on completed trials."""
        if self.study is None:
            return total_trials
        
        completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, total_trials - completed)
        
        print(f"Completed trials: {completed}/{total_trials}")
        print(f"Remaining trials: {remaining}")
        
        return remaining
    
    def _save_trial_results(self, trial_number: int, params: Dict, graph_results: Dict, objective_value: float):
        """Save trial results to JSON (excluding non-serializable objects)."""
        trial_dir = self.results_path / f"trial_{trial_number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean graph results (remove non-JSON serializable data)
        clean_results = {
            name: {k: v for k, v in data.items() if k not in ['graph', 'solution']}
            for name, data in graph_results.items()
        }
        
        output = {
            "trial_number": trial_number,
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
            "objective_value": objective_value,
            "total_color_count": objective_value,
            "graphs": clean_results
        }
        
        with open(trial_dir / "trial_results.json", 'w') as f:
            json.dump(output, f, indent=2)
    
    def optimize(
        self,
        objective_func: Callable,
        param_config: Dict[str, Dict[str, Any]],
        aco_class: type,
        n_trials: int = 100,
        aco_fixed_params: Optional[Dict] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Objective function
            param_config: Parameter search space configuration
            aco_class: ACO class to instantiate
            n_trials: Total number of trials
            aco_fixed_params: Fixed parameters for ACO
            timeout: Time limit in seconds
            n_jobs: Number of parallel jobs
            show_progress_bar: Show Optuna progress bar
            
        Returns:
            Best parameters dictionary
        """
        # Create or load study
        if self.study is None:
            self.create_or_load_study()
        
        # Calculate remaining trials
        remaining = self._get_remaining_trials(n_trials)
        
        if remaining <= 0:
            print("All trials already completed!")
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            return self.best_params
        
        # Create wrapped objective with parameter suggestions
        def wrapped_objective(trial: optuna.Trial) -> float:
            suggested_params = {}
            for param_name, param_spec in param_config.items():
                param_type = param_spec['type']
                
                if param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, param_spec['low'], param_spec['high']
                    )
                elif param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, param_spec['low'], param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name, param_spec['choices']
                    )
            
            return objective_func(
                trial=trial,
                params=suggested_params,
                aco_class=aco_class,
                fixed_params=aco_fixed_params or {}
            )
        
        # Define callback to save trial results and generate visualizations
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                graph_results = trial.user_attrs.get('graph_results', {})
                params = trial.params
                
                # Save JSON results
                self._save_trial_results(trial.number, params, graph_results, trial.value)
                
                # Generate visualizations with graph data from global storage
                graph_data_for_viz = _trial_graph_viz_data.get(trial.number, {})
                if graph_data_for_viz:
                    # Trial figures go in results/trial_XXXX/ folder (same as trial data)
                    trial_dir = self.results_path / f"trial_{trial.number:04d}"
                    # Merge viz data
                    graph_results_with_viz = {
                        name: {**data, **graph_data_for_viz.get(name, {})}
                        for name, data in graph_results.items()
                    }
                    generate_trial_visualizations(trial.number, graph_results_with_viz, trial_dir)
                
                # Update study summary after each trial (silent)
                self.save_study_summary(silent=True)
                
                # Clean up
                _trial_graph_viz_data.pop(trial.number, None)
        
        # Run optimization
        print(f"\nStarting optimization with {remaining} trials...")
        self.study.optimize(
            wrapped_objective,
            n_trials=remaining,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=[trial_callback]
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"\nOptimization complete!")
        print(f"Best value: {self.best_value}")
        print(f"Best parameters: {self.best_params}")
        
        # Save comprehensive study summary (with message)
        summary_path = self.save_study_summary(silent=False)
        
        # Generate study-level visualizations
        self.generate_plots(recreate=True)
        
        return self.best_params
    
    def generate_plots(self, recreate: bool = False):
        """Generate all study-level visualization plots."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        print("\nGenerating visualization plots...")
        
        plots = [
            (self.figures_path / "history.png", lambda: plot_study_history(self.study, self.figures_path)),
            (self.figures_path / "importances.png", lambda: plot_param_importances(self.study, self.figures_path)),
            (self.figures_path / "slice.png", lambda: plot_slice(self.study, self.figures_path)),
            (self.figures_path / "timeline.png", lambda: plot_timeline(self.study, self.figures_path)),
        ]
        
        for filepath, plot_func in plots:
            if recreate or not filepath.exists():
                plot_func()
            else:
                print(f"{filepath.stem} already exists: {filepath.name}")
        
        print("All plots generated successfully.\n")
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get trials as a pandas DataFrame."""
        if self.study is None:
            raise ValueError("No study available")
        return self.study.trials_dataframe()
    
    def save_trials_csv(self, filename: str = None) -> str:
        """Save all trials to CSV file."""
        if self.study is None:
            raise ValueError("No study to save")
        
        filename = filename or f"{self.study_name}_trials.csv"
        filepath = self.study_folder / filename
        
        df = self.get_trials_dataframe()
        df.to_csv(filepath, index=False)
        
        print(f"Trials saved to: {filepath}")
        return str(filepath)
    
    def save_study_summary(self, silent: bool = False) -> str:
        """Save comprehensive study summary as JSON.
        
        Args:
            silent: If True, don't print the save location message
        """
        if self.study is None:
            raise ValueError("No study to save")
        
        # Collect all trials information
        trials_info = []
        for trial in self.study.trials:
            trial_info = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value if trial.value is not None else None,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds() 
                                   if trial.datetime_complete and trial.datetime_start else None,
                "parameters": trial.params,
                "user_attrs": {k: v for k, v in trial.user_attrs.items() if k not in ['graph_results']},
                "system_attrs": trial.system_attrs
            }
            trials_info.append(trial_info)
        
        # Create comprehensive summary
        summary = {
            "study_name": self.study_name,
            "direction": self.direction,
            "creation_time": datetime.now().isoformat(),
            "n_trials": len(self.study.trials),
            "best_trial": {
                "number": self.study.best_trial.number,
                "value": self.study.best_value,
                "parameters": self.study.best_params,
                "datetime_start": self.study.best_trial.datetime_start.isoformat() if self.study.best_trial.datetime_start else None,
                "datetime_complete": self.study.best_trial.datetime_complete.isoformat() if self.study.best_trial.datetime_complete else None
            },
            "trials": trials_info,
            "study_statistics": {
                "complete_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
                "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "running_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.RUNNING])
            },
            "storage_info": {
                "type": "JournalStorage",
                "journal_file": str(self.journal_file),
                "note": "Study can be resumed by loading from this journal file"
            }
        }
        
        # Save to JSON file
        summary_file = self.study_folder / f"{self.study_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if not silent:
            print(f"Study summary saved to: {summary_file}")
        return str(summary_file)
