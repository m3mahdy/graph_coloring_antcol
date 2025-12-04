#!/usr/bin/env python3
"""
Generate parameter slice plot with 5 columns and 2 rows layout.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def generate_slice_plot(json_path, output_path):
    """Generate slice plot for hyperparameter analysis."""
    
    print(f"Reading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trials = data['trials']
    print(f"Processing {len(trials)} trials")
    
    # Extract parameters and values
    params = {}
    for trial in trials:
        for param_name, param_value in trial['parameters'].items():
            if param_name not in params:
                params[param_name] = {'values': [], 'objectives': []}
            params[param_name]['values'].append(param_value)
            params[param_name]['objectives'].append(trial['value'])
    
    # Parameter names in order
    param_names = ['iterations', 'alpha', 'beta', 'rho', 'ant_count', 'Q', 'patience']
    param_labels = {
        'iterations': 'Iterations',
        'alpha': 'Alpha (α)',
        'beta': 'Beta (β)',
        'rho': 'Rho (ρ)',
        'ant_count': 'Ant Count',
        'Q': 'Q',
        'patience': 'Patience'
    }
    
    # Create figure with 4 columns and 2 rows
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Plot each parameter
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        
        if param_name in params:
            x_vals = params[param_name]['values']
            y_vals = params[param_name]['objectives']
            
            # Scatter plot with RED points (matching timeline style)
            ax.scatter(x_vals, y_vals, alpha=0.7, s=60, c='#e74c3c', edgecolors='black', linewidth=0.8)
            
            # Try to fit a trend line if enough data points
            if len(x_vals) > 3:
                try:
                    # Sort by x values for trend line
                    sorted_indices = np.argsort(x_vals)
                    x_sorted = np.array(x_vals)[sorted_indices]
                    y_sorted = np.array(y_vals)[sorted_indices]
                    
                    # Fit polynomial (degree 2)
                    z = np.polyfit(x_sorted, y_sorted, 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                    y_trend = p(x_trend)
                    # Use blue trend line for better contrast with red points
                    ax.plot(x_trend, y_trend, 'b-', linewidth=2.5, alpha=0.6)
                except:
                    pass
            
            ax.set_xlabel(param_labels[param_name], fontsize=12, fontweight='bold')
            ax.set_ylabel('Colors' if idx % 4 == 0 else '', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            
            # Increase tick label font size for clarity
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Format x-axis for better readability
            if param_name in ['rho', 'patience', 'Q']:
                ax.ticklabel_format(style='plain', axis='x')
    
    # Hide empty subplot (we have 7 parameters, 1 spot will be empty)
    for idx in range(len(param_names), 8):
        axes[idx].axis('off')
    
    # Adjust layout - no main title, just tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'\n✓ Slice plot saved: {output_path}')
    print(f'✓ Layout: 4 columns × 2 rows')
    print(f'✓ Parameters: {len(param_names)}\n')
    
    plt.close()


if __name__ == '__main__':
    json_file = '../data/studies/aco_study_limited_dataset_20251130_195639/aco_study_limited_dataset_20251130_195639_summary.json'
    output_file = '../figures/slice.png'
    
    try:
        generate_slice_plot(json_file, output_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
