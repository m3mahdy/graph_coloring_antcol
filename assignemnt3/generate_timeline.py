#!/usr/bin/env python3
"""
Generate timeline figure for Optuna hyperparameter tuning results.
Reads trial data from JSON and creates a clear timeline visualization.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

def generate_timeline(json_path, output_path):
    """Generate timeline figure from trial data."""
    
    # Read the JSON file
    print(f"Reading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trials = data['trials']
    print(f"Found {len(trials)} trials")
    
    # Extract trial data
    trial_numbers = [t['trial_number'] for t in trials]
    start_times = [datetime.fromisoformat(t['datetime_start']) for t in trials]
    end_times = [datetime.fromisoformat(t['datetime_complete']) for t in trials]
    best_colors = [t['value'] for t in trials]
    
    # Print date range for verification
    min_date = min(start_times)
    max_date = max(end_times)
    print(f"Date range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Dates should show: {min_date.strftime('%d %b')}, {max_date.strftime('%d %b')}")
    
    # Calculate durations in hours
    durations = [(end - start).total_seconds() / 3600 for start, end in zip(start_times, end_times)]
    
    # Create color map based on performance
    colors = []
    for bc in best_colors:
        if bc <= 205:
            colors.append('#2ecc71')  # Green
        elif bc <= 220:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red
    
    # Create figure - wider for horizontal space
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot horizontal bars
    y_positions = range(len(trial_numbers))
    bars = ax.barh(y_positions, durations, left=[mdates.date2num(st) for st in start_times], 
                   color=colors, edgecolor='black', linewidth=0.5, height=0.85)
    
    # Add color count text on each bar - ALL BLACK
    for i, (bar, color_count) in enumerate(zip(bars, best_colors)):
        width = bar.get_width()
        x_pos = bar.get_x() + width / 2
        y_pos = bar.get_y() + bar.get_height() / 2
        
        ax.text(x_pos, y_pos, str(int(color_count)), 
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='black')
    
    # Highlight the best trial (Trial 19)
    best_trial_idx = 19
    ax.plot(mdates.date2num(start_times[best_trial_idx]), best_trial_idx, 
            marker='*', markersize=18, color='gold', markeredgecolor='black', 
            markeredgewidth=2, zorder=10)
    
    # Format x-axis with 8-hour intervals
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    
    # STATIC date formatter - hardcoded correct dates from the actual trial data
    # Data runs from: 2025-11-30 19:56 to 2025-12-02 17:57
    def static_date_formatter(x, pos):
        dt = mdates.num2date(x)
        hour = dt.hour
        day = dt.day
        month_name = dt.strftime('%b')
        
        # Explicitly map dates to prevent any formatting issues
        # November 30, December 1, December 2
        if day == 30 and month_name == 'Nov':
            date_str = "30 Nov"
        elif day == 1 and month_name == 'Dec':
            date_str = "1 Dec"
        elif day == 2 and month_name == 'Dec':
            date_str = "2 Dec"
        else:
            # Fallback - use the actual day and month from datetime
            date_str = f"{day} {month_name}"
        
        return f"{hour:02d}:00\n{date_str}"
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(static_date_formatter))
    
    # Larger font for x-axis labels
    ax.tick_params(axis='x', labelsize=11, pad=8)
    
    # Set labels and title
    ax.set_xlabel('Time (Hour:Minute, Day Month)', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('Trial Number', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Optuna Hyperparameter Tuning Timeline', fontsize=15, fontweight='bold', pad=20)
    
    # Set y-axis - show every other trial for clarity
    ax.set_yticks(y_positions[::2])
    ax.set_yticklabels([f'{i}' for i in trial_numbers[::2]], fontsize=9)
    ax.set_ylim(-0.5, len(trial_numbers) - 0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='≤205 colors'),
        Patch(facecolor='#f39c12', edgecolor='black', label='206-220 colors'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='>220 colors'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                   markersize=14, markeredgecolor='black', markeredgewidth=2, 
                   label='Best: T19 (200)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
    
    # Add subtitle
    fig.text(0.5, 0.02, 
             '40 trials | 6 workers | 46h wall-clock | Numbers on bars show color count', 
             ha='center', fontsize=10, style='italic', color='#555555')
    
    # Grid
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.1, axis='y', linestyle=':', linewidth=0.5)
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'✓ Timeline saved to: {output_path}')
    print(f'✓ X-axis format: "HH:MM" on line 1, "DD Mon" on line 2')
    print(f'✓ Date verification: {min_date.day} {min_date.strftime("%b")} to {max_date.day} {max_date.strftime("%b")}')
    plt.close()


if __name__ == '__main__':
    json_file = '../data/studies/aco_study_limited_dataset_20251130_195639/aco_study_limited_dataset_20251130_195639_summary.json'
    output_file = '../figures/timeline.png'
    
    try:
        generate_timeline(json_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
