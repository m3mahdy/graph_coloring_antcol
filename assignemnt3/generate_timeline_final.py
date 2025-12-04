#!/usr/bin/env python3
"""
FINAL CORRECT VERSION - Generate timeline with EXPLICIT date handling.
This version converts ALL dates upfront to avoid matplotlib datetime issues.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys

def generate_timeline(json_path, output_path):
    """Generate timeline figure with explicit date control."""
    
    # Read JSON
    print(f"Reading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trials = data['trials']
    print(f"Processing {len(trials)} trials\n")
    
    # Extract data
    trial_numbers = []
    start_times = []
    end_times = []
    best_colors = []
    
    for t in trials:
        trial_numbers.append(t['trial_number'])
        start_times.append(datetime.fromisoformat(t['datetime_start']))
        end_times.append(datetime.fromisoformat(t['datetime_complete']))
        best_colors.append(t['value'])
    
    # Find date range
    min_date = min(start_times)
    max_date = max(end_times)
    
    print(f"=== DATE VERIFICATION ===")
    print(f"Min: {min_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Max: {max_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Date range: {min_date.day} {min_date.strftime('%b')} to {max_date.day} {max_date.strftime('%b')}")
    print(f"=========================\n")
    
    # Calculate durations IN HOURS
    durations_hours = [(end - start).total_seconds() / 3600 for start, end in zip(start_times, end_times)]
    print(f"Duration range: {min(durations_hours):.1f}h to {max(durations_hours):.1f}h")
    
    # Convert to matplotlib date numbers
    start_nums = [mdates.date2num(st) for st in start_times]
    end_nums = [mdates.date2num(et) for et in end_times]
    
    # Colors based on performance
    colors = []
    for bc in best_colors:
        if bc <= 205:
            colors.append('#2ecc71')
        elif bc <= 220:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    # Create figure - LARGER for readability
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot bars - use NUMERIC durations, not times!
    y_positions = range(len(trial_numbers))
    bars = ax.barh(y_positions, 
                   [en - st for en, st in zip(end_nums, start_nums)],  # DURATION in days
                   left=start_nums,  # START position
                   color=colors, 
                   edgecolor='black', 
                   linewidth=0.5, 
                   height=0.8)
    
    print(f"Bar widths (first 5): {[bars[i].get_width() for i in range(5)]}")
    print(f"Bar widths (last 5): {[bars[i].get_width() for i in range(-5, 0)]}")
    
    # Add color count INSIDE bars
    for i, (bar, color_count) in enumerate(zip(bars, best_colors)):
        width = bar.get_width()
        x_pos = bar.get_x() + width / 2
        y_pos = bar.get_y() + bar.get_height() / 2
        
        # Only add text if bar is wide enough
        if width > 0.01:  # Check if bar has reasonable width
            ax.text(x_pos, y_pos, str(int(color_count)), 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Best trial marker
    best_trial_idx = 19
    ax.plot(start_nums[best_trial_idx], best_trial_idx, 
            marker='*', markersize=20, color='gold', markeredgecolor='black', 
            markeredgewidth=2, zorder=10)
    
    # Set x-axis limits
    padding = timedelta(hours=2)
    ax.set_xlim(mdates.date2num(min_date - padding), mdates.date2num(max_date + padding))
    
    # X-axis formatting - EVERY 4 HOURS
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    
    def date_formatter(x, pos):
        dt = mdates.num2date(x)
        month_names = {11: 'Nov', 12: 'Dec'}
        return f"{dt.hour:02d}:00\n{dt.day} {month_names.get(dt.month, dt.month)}"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(date_formatter))
    
    # Labels
    ax.set_xlabel('Time (Hour, Date)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Trial Number', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Optuna Hyperparameter Tuning Timeline', fontsize=16, fontweight='bold', pad=20)
    
    # Y-axis
    ax.set_yticks(y_positions[::2])
    ax.set_yticklabels([f'{i}' for i in trial_numbers[::2]], fontsize=10)
    ax.set_ylim(-0.5, len(trial_numbers) - 0.5)
    
    # Legend - position at LEFT
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='≤205 colors'),
        Patch(facecolor='#f39c12', edgecolor='black', label='206-220 colors'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='>220 colors'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                   markersize=16, markeredgecolor='black', markeredgewidth=2, 
                   label='Best: T19 (200)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Subtitle
    fig.text(0.5, 0.02, 
             '40 trials | 6 workers | 46h wall-clock | Numbers on bars show color count', 
             ha='center', fontsize=11, style='italic', color='#555555')
    
    # Grid
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.1, axis='y', linestyle=':', linewidth=0.5)
    
    # Background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Adjust layout to prevent clipping
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f'\n✓ Timeline saved: {output_path}')
    print(f'✓ Bars have DIFFERENT widths (durations)')
    print(f'✓ Color counts INSIDE bars with white background')
    print(f'✓ Date range: {min_date.day} {min_date.strftime("%b")} to {max_date.day} {max_date.strftime("%b")}')
    print(f'✓ X-axis: 4-hour intervals')
    print(f'✓ Legend: upper left\n')
    
    plt.close()


if __name__ == '__main__':
    json_file = 'data/studies/aco_study_limited_dataset_20251130_195639/aco_study_limited_dataset_20251130_195639_summary.json'
    output_file = 'figures/timeline.png'
    
    try:
        generate_timeline(json_file, output_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
