"""
Optuna Hyperparameter Tuner for ACO Graph Coloring
Provides flexible hyperparameter optimization with study persistence, resumption, and visualization.
"""

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import json
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


class OptunaACOTuner:
    """
    Optuna-based hyperparameter tuner for ACO algorithms.
    
    Features:
    - JSON-based study persistence and resumption
    - Automatic tracking of remaining trials
    - Integration with GraphDataLoader
    - Per-trial result saving for each graph
    - Visualization of optimization progress
    - Best parameter saving to JSON
    """
    
    def __init__(
        self,
        study_name: str,
        data_root: str,
        direction: str = "minimize"
    ):
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
        
        # Setup JSON storage (in study folder)
        self.journal_file = self.study_folder / f"{study_name}.log"
        self.storage = JournalStorage(JournalFileStorage(str(self.journal_file)))
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        self.best_value: Optional[float] = None
    
    def create_or_load_study(self, load_if_exists: bool = True) -> optuna.Study:
        """
        Create a new study or load an existing one from JSON storage.
        
        Args:
            load_if_exists: If True, load existing study; if False, create new
            
        Returns:
            Optuna Study object
        """
        try:
            self.study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )
            print(f"Loaded existing study '{self.study_name}' with {len(self.study.trials)} completed trials")
        except KeyError:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                load_if_exists=False
            )
            print(f"Created new study '{self.study_name}'")
        
        return self.study
    
    def _get_remaining_trials(self, total_trials: int) -> int:
        """
        Calculate remaining trials based on completed trials.
        
        Args:
            total_trials: Total number of trials to run
            
        Returns:
            Number of remaining trials
        """
        if self.study is None:
            return total_trials
        
        completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, total_trials - completed)
        
        print(f"Completed trials: {completed}/{total_trials}")
        print(f"Remaining trials: {remaining}")
        
        return remaining
    
    def _save_trial_results(self, trial_number: int, graph_results: Dict[str, Dict]):
        """
        Save results for a specific trial.
        
        Args:
            trial_number: Trial number
            graph_results: Dictionary mapping graph name to result dict
                          {graph_name: {color_count: int, conflict_count: int, ...}}
        """
        trial_dir = self.results_path / f"trial_{trial_number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = trial_dir / "results.json"
        
        output = {
            "trial_number": trial_number,
            "timestamp": datetime.now().isoformat(),
            "graphs": graph_results
        }
        
        with open(results_file, 'w') as f:
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
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Objective function with signature:
                           func(trial, params, data_loader, aco_class, fixed_params) -> float
                           The function should iterate over graphs from data_loader and return a score
            param_config: Dictionary defining parameter search space.
                         Format: {
                             'param_name': {
                                 'type': 'int'/'float'/'categorical',
                                 'low': min_value,  # for int/float
                                 'high': max_value,  # for int/float
                                 'choices': [...]  # for categorical
                             }
                         }
            aco_class: ACO class to instantiate (e.g., ACOGraphColoring)
            n_trials: Total number of trials (including completed ones)
            aco_fixed_params: Fixed parameters for ACO (e.g., num_colors)
            timeout: Time limit in seconds for optimization
            n_jobs: Number of parallel jobs (1 for sequential, -1 for all cores)
            show_progress_bar: Show Optuna progress bar
            
        Returns:
            Completed Optuna Study object
        """
        # Create or load study
        if self.study is None:
            self.create_or_load_study(load_if_exists=True)
        
        # Calculate remaining trials
        remaining = self._get_remaining_trials(n_trials)
        
        if remaining <= 0:
            print("All trials already completed!")
            return self.study
        
        # Create wrapped objective with parameter suggestions
        def wrapped_objective(trial: optuna.Trial) -> float:
            # Suggest parameters based on config
            suggested_params = {}
            for param_name, param_spec in param_config.items():
                param_type = param_spec['type']
                
                if param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high']
                    )
                elif param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec['choices']
                    )
            
            # Call user's objective function with data_loader
            score = objective_func(
                trial=trial,
                params=suggested_params,
                data_loader=self.data_loader,
                aco_class=aco_class,
                fixed_params=aco_fixed_params if aco_fixed_params else {}
            )
            
            return score
        
        # Run optimization
        print(f"\nStarting optimization with {remaining} trials...")
        self.study.optimize(
            wrapped_objective,
            n_trials=remaining,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"\nOptimization complete!")
        print(f"Best value: {self.best_value}")
        print(f"Best parameters: {self.best_params}")
        
        # Generate visualization plots
        self.generate_plots(recreate=True)
        
        return self.best_params
    
    def generate_plots(self, recreate: bool = False):
        """
        Generate all visualization plots for the study.
        Creates or recreates all plots in the figures folder.
        
        Args:
            recreate: If True, regenerate all plots even if they exist.
                     If False, only generate missing plots.
        """
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        print("\nGenerating visualization plots...")
        
        # Define plot filenames
        history_file = self.figures_path / f"{self.study_name}_history.png"
        importances_file = self.figures_path / f"{self.study_name}_importances.png"
        parallel_file = self.figures_path / f"{self.study_name}_parallel.png"
        
        # Generate plots based on recreate flag
        if recreate or not history_file.exists():
            self._plot_optimization_history()
        else:
            print(f"Optimization history already exists: {history_file}")
        
        if recreate or not importances_file.exists():
            self._plot_param_importances()
        else:
            print(f"Parameter importances already exists: {importances_file}")
        
        if recreate or not parallel_file.exists():
            self._plot_parallel_coordinate()
        else:
            print(f"Parallel coordinate plot already exists: {parallel_file}")
        
        print("All plots generated successfully.\n")
    
    def _save_best_params(self, filename: str = None) -> str:
        """
        Save best parameters to JSON file.
        
        Args:
            filename: Custom filename. If None, uses study_name_best_params.json
            
        Returns:
            Path to saved file
        """
        if self.study is None or self.best_params is None:
            raise ValueError("No study results to save. Run optimization first.")
        
        if filename is None:
            filename = f"{self.study_name}_best_params.json"
        
        filepath = self.studies_path / filename
        
        output = {
            "study_name": self.study_name,
            "dataset_name": self.dataset_name,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": len(self.study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Best parameters saved to: {filepath}")
        return str(filepath)
    
    def _plot_optimization_history(self, filename: str = None, show: bool = False):
        """
        Plot optimization history showing best value over trials.
        
        Args:
            filename: Output filename. If None, uses study_name_history.png
            show: Whether to display the plot
        """
        if self.study is None:
            raise ValueError("No study to plot. Run optimization first.")
        
        if filename is None:
            filename = f"{self.study_name}_history.png"
        
        filepath = self.figures_path / filename
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"Optimization history saved to: {filepath}")
    
    def _plot_param_importances(self, filename: str = None, show: bool = False):
        """
        Plot parameter importance.
        
        Args:
            filename: Output filename. If None, uses study_name_importances.png
            show: Whether to display the plot
        """
        if self.study is None:
            raise ValueError("No study to plot. Run optimization first.")
        
        if len(self.study.trials) < 2:
            print("Not enough trials to compute parameter importances")
            return
        
        if filename is None:
            filename = f"{self.study_name}_importances.png"
        
        filepath = self.figures_path / filename
        
        try:
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            print(f"Parameter importances saved to: {filepath}")
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
    
    def _plot_parallel_coordinate(self, filename: str = None, show: bool = False):
        """
        Plot parallel coordinate plot of parameters.
        
        Args:
            filename: Output filename. If None, uses study_name_parallel.png
            show: Whether to display the plot
        """
        if self.study is None:
            raise ValueError("No study to plot. Run optimization first.")
        
        if filename is None:
            filename = f"{self.study_name}_parallel.png"
        
        filepath = self.figures_path / filename
        
        try:
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            print(f"Parallel coordinate plot saved to: {filepath}")
        except Exception as e:
            print(f"Could not plot parallel coordinate: {e}")
    
    def _get_trials_dataframe(self) -> pd.DataFrame:
        """
        Get trials as a pandas DataFrame.
        
        Returns:
            DataFrame with trial information
        """
        if self.study is None:
            raise ValueError("No study available")
        
        return self.study.trials_dataframe()
    
    def _save_trials_csv(self, filename: str = None) -> str:
        """
        Save all trials to CSV file.
        
        Args:
            filename: Output filename. If None, uses study_name_trials.csv
            
        Returns:
            Path to saved file
        """
        if self.study is None:
            raise ValueError("No study to save")
        
        if filename is None:
            filename = f"{self.study_name}_trials.csv"
        
        filepath = self.studies_path / filename
        
        df = self._get_trials_dataframe()
        df.to_csv(filepath, index=False)
        
        print(f"Trials saved to: {filepath}")
        return str(filepath)
    
    def save_trial_graph_results(self, trial_number: int, graph_results: Dict[str, Dict]):
        """
        Public method to save results for each graph in a trial.
        Call this from your objective function.
        
        Args:
            trial_number: Trial number
            graph_results: Dictionary mapping graph name to result dict
                          Example: {
                              'dsjc250.5': {
                                  'color_count': 28,
                                  'conflict_count': 0,
                                  'execution_time': 1.23
                              }
                          }
        """
        self._save_trial_results(trial_number, graph_results)

