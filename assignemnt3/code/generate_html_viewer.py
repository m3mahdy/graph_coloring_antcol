"""
Generate an interactive HTML viewer for ACO visualizations.
Shows all ants and pheromone heatmap for each step, with navigation controls.
"""
import json
from pathlib import Path
import sys


def generate_html_viewer(viz_dir):
    """
    Generate an interactive HTML viewer for ACO visualization results.
    
    Args:
        viz_dir: Path to the visualization directory (e.g., aco_visualizations/test_20251130_001635)
    """
    viz_path = Path(viz_dir)
    
    if not viz_path.exists():
        print(f"Error: Directory {viz_dir} does not exist")
        return
    
    # Get parent directory to find all test runs
    parent_dir = viz_path.parent
    all_test_dirs = sorted([d for d in parent_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('test_')], 
                          reverse=True)
    current_test = viz_path.name
    
    # Build list of available tests
    available_tests = []
    for test_dir in all_test_dirs:
        available_tests.append({
            'name': test_dir.name,
            'path': test_dir.name
        })
    
    # Discover structure
    iterations = sorted([d for d in viz_path.iterdir() if d.is_dir() and d.name.startswith('iteration_')])
    
    if not iterations:
        print("No iteration directories found")
        return
    
    # Build data structure
    data = {}
    max_steps = 0
    num_ants = 0
    
    for iter_dir in iterations:
        iter_num = int(iter_dir.name.split('_')[1])
        ant_dirs = sorted([d for d in iter_dir.iterdir() if d.is_dir() and d.name.startswith('ant_')])
        num_ants = max(num_ants, len(ant_dirs))
        
        # Get pheromone heatmap
        pheromone_path = iter_dir / 'pheromone.png'
        best_global_path = iter_dir / 'best_global.png'
        
        # Get steps from first ant to determine max steps
        if ant_dirs:
            first_ant = ant_dirs[0]
            steps = sorted([f for f in first_ant.iterdir() if f.name.startswith('step_') and f.suffix == '.png'])
            max_steps = max(max_steps, len(steps))
        
        data[iter_num] = {
            'pheromone': str(pheromone_path.relative_to(viz_path)) if pheromone_path.exists() else None,
            'best_global': str(best_global_path.relative_to(viz_path)) if best_global_path.exists() else None,
            'ants': {}
        }
        
        for ant_dir in ant_dirs:
            ant_num = int(ant_dir.name.split('_')[1])
            steps = sorted([f for f in ant_dir.iterdir() if f.name.startswith('step_') and f.suffix == '.png'])
            final = ant_dir / 'final_solution.png'
            
            data[iter_num]['ants'][ant_num] = {
                'steps': [str(s.relative_to(viz_path)) for s in steps],
                'final': str(final.relative_to(viz_path)) if final.exists() else None
            }
    
    # Build test selection options
    test_options = '\n'.join([
        f'                    <option value="{t["path"]}" {"selected" if t["path"] == current_test else ""}>{t["name"]}</option>'
        for t in available_tests
    ])
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ACO Visualization Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            padding: 10px;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 10px;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
        }}
        
        h1 {{
            font-size: 1.3em;
            margin-bottom: 3px;
        }}
        
        header p {{
            font-size: 0.75em;
            color: #aaa;
            margin: 0;
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin: 8px 0;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        button {{
            background: #0066cc;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.3s;
        }}
            transition: background 0.3s;
        }}
        
        button:hover {{
            background: #0052a3;
        }}
        
        button:disabled {{
            background: #555;
            cursor: not-allowed;
        }}
        
        label {{
            font-weight: bold;
            margin-right: 3px;
            font-size: 12px;
        }}
        
        select, input[type="range"] {{
            padding: 4px 6px;
            border-radius: 4px;
            border: 1px solid #555;
            background: #333;
            color: white;
            font-size: 12px;
        }}
        
        input[type="range"] {{
            width: 150px;
        }}
        
        .info {{
            text-align: center;
            font-size: 13px;
            margin: 5px 0;
            padding: 6px;
            background: #2a2a2a;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .viewer {{
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 15px;
            margin-top: 10px;
            align-items: start;
        }}
        
        .pheromone-section {{
            position: sticky;
            top: 10px;
        }}
        
        .ants-section {{
            display: grid;
            grid-template-columns: repeat({num_ants}, 1fr);
            gap: 12px;
            width: 100%;
        }}
        
        .image-card {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .image-card h3 {{
            margin-bottom: 8px;
            text-align: center;
            color: #66b3ff;
            font-size: 0.85em;
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
        }}
        
        .pheromone-section .image-card {{
            padding: 8px;
        }}
        
        .pheromone-section .image-card h3 {{
            font-size: 0.75em;
            margin-bottom: 5px;
        }}
        
        .keyboard-hints {{
            text-align: center;
            margin-top: 10px;
            padding: 8px;
            background: #2a2a2a;
            border-radius: 4px;
            font-size: 11px;
            color: #aaa;
        }}
        
        .keyboard-hints kbd {{
            background: #444;
            padding: 2px 5px;
            border-radius: 3px;
            border: 1px solid #666;
            font-family: monospace;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐜 ACO Graph Coloring Visualization</h1>
            <p>Interactive viewer for Ant Colony Optimization step-by-step visualization</p>
        </header>
        
        <div class="controls">
            <div class="control-group">
                <label for="testSelect">Test Run:</label>
                <select id="testSelect">
{test_options}
                </select>
            </div>
            
            <div class="control-group">
                <label for="iterationSelect">Iteration:</label>
                <select id="iterationSelect"></select>
            </div>
            
            <div class="control-group">
                <label for="stepSlider">Step:</label>
                <input type="range" id="stepSlider" min="0" max="{max_steps}" value="0">
                <span id="stepLabel">0 / {max_steps}</span>
            </div>
            
            <div class="control-group">
                <button id="prevBtn">◀ Previous</button>
                <button id="playBtn">▶ Play</button>
                <button id="nextBtn">Next ▶</button>
            </div>
            
            <div class="control-group">
                <label for="speedSelect">Speed:</label>
                <select id="speedSelect">
                    <option value="2000">Slow</option>
                    <option value="1000" selected>Normal</option>
                    <option value="500">Fast</option>
                    <option value="200">Very Fast</option>
                </select>
            </div>
        </div>
        
        <div class="info" id="infoBar">
            Loading...
        </div>
        
        <div class="viewer" id="viewer">
        </div>
        
        <div class="keyboard-hints">
            <strong>Keyboard Shortcuts:</strong>
            <kbd>←</kbd> Previous Step |
            <kbd>→</kbd> Next Step |
            <kbd>Space</kbd> Play/Pause |
            <kbd>↑</kbd> Previous Iteration |
            <kbd>↓</kbd> Next Iteration
        </div>
    </div>
    
    <script>
        const data = {json.dumps(data, indent=8)};
        const maxSteps = {max_steps};
        const numAnts = {num_ants};
        
        let currentIteration = 0;
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        
        const testSelect = document.getElementById('testSelect');
        const iterationSelect = document.getElementById('iterationSelect');
        const stepSlider = document.getElementById('stepSlider');
        const stepLabel = document.getElementById('stepLabel');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const playBtn = document.getElementById('playBtn');
        const speedSelect = document.getElementById('speedSelect');
        const infoBar = document.getElementById('infoBar');
        const viewer = document.getElementById('viewer');
        
        // Handle test selection change
        testSelect.addEventListener('change', (e) => {{
            const selectedTest = e.target.value;
            // Navigate to the viewer in the selected test directory
            window.location.href = `../${{selectedTest}}/viewer.html`;
        }});
        
        // Initialize iteration dropdown
        const iterations = Object.keys(data).map(Number).sort((a, b) => a - b);
        iterations.forEach(iter => {{
            const option = document.createElement('option');
            option.value = iter;
            option.textContent = `Iteration ${{iter}}`;
            iterationSelect.appendChild(option);
        }});
        
        function updateDisplay() {{
            const iterData = data[currentIteration];
            if (!iterData) return;
            
            // Update info bar
            infoBar.textContent = `Iteration ${{currentIteration}} - Step ${{currentStep + 1}} / ${{maxSteps}}`;
            stepLabel.textContent = `${{currentStep}} / ${{maxSteps - 1}}`;
            stepSlider.value = currentStep;
            iterationSelect.value = currentIteration;
            
            // Clear viewer
            viewer.innerHTML = '';
            
            // Create pheromone section
            const pheromoneSection = document.createElement('div');
            pheromoneSection.className = 'pheromone-section';
            
            // Show pheromone heatmap
            if (iterData.pheromone) {{
                const card = document.createElement('div');
                card.className = 'image-card';
                card.innerHTML = `
                    <h3>🔥 Pheromone Trails - Iteration ${{currentIteration}}</h3>
                    <img src="${{iterData.pheromone}}" alt="Pheromone heatmap">
                `;
                pheromoneSection.appendChild(card);
            }}
            viewer.appendChild(pheromoneSection);
            
            // Create ants section
            const antsSection = document.createElement('div');
            antsSection.className = 'ants-section';
            
            // Show all ants for current step
            Object.keys(iterData.ants).sort((a, b) => Number(a) - Number(b)).forEach(antNum => {{
                const ant = iterData.ants[antNum];
                const card = document.createElement('div');
                card.className = 'image-card';
                
                let imagePath;
                let title;
                
                if (currentStep < ant.steps.length) {{
                    imagePath = ant.steps[currentStep];
                    title = `🐜 Ant ${{antNum}} - Step ${{currentStep + 1}}`;
                }} else if (ant.final) {{
                    imagePath = ant.final;
                    title = `🐜 Ant ${{antNum}} - Final Solution`;
                }} else {{
                    return;
                }}
                
                card.innerHTML = `
                    <h3>${{title}}</h3>
                    <img src="${{imagePath}}" alt="Ant ${{antNum}} step ${{currentStep}}">
                `;
                antsSection.appendChild(card);
            }});
            viewer.appendChild(antsSection);
            
            // Update button states
            prevBtn.disabled = currentIteration === iterations[0] && currentStep === 0;
            nextBtn.disabled = currentIteration === iterations[iterations.length - 1] && currentStep >= maxSteps - 1;
        }}
        
        function nextStep() {{
            if (currentStep < maxSteps - 1) {{
                currentStep++;
            }} else if (currentIteration < iterations[iterations.length - 1]) {{
                currentIteration = iterations[iterations.indexOf(currentIteration) + 1];
                currentStep = 0;
            }} else {{
                stopPlay();
                return;
            }}
            updateDisplay();
        }}
        
        function prevStep() {{
            if (currentStep > 0) {{
                currentStep--;
            }} else if (currentIteration > iterations[0]) {{
                currentIteration = iterations[iterations.indexOf(currentIteration) - 1];
                currentStep = maxSteps - 1;
            }} else {{
                return;
            }}
            updateDisplay();
        }}
        
        function startPlay() {{
            if (isPlaying) return;
            isPlaying = true;
            playBtn.textContent = '⏸ Pause';
            const speed = parseInt(speedSelect.value);
            playInterval = setInterval(nextStep, speed);
        }}
        
        function stopPlay() {{
            isPlaying = false;
            playBtn.textContent = '▶ Play';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        function togglePlay() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}
        
        // Event listeners
        nextBtn.addEventListener('click', () => {{
            stopPlay();
            nextStep();
        }});
        
        prevBtn.addEventListener('click', () => {{
            stopPlay();
            prevStep();
        }});
        
        playBtn.addEventListener('click', togglePlay);
        
        stepSlider.addEventListener('input', (e) => {{
            stopPlay();
            currentStep = parseInt(e.target.value);
            updateDisplay();
        }});
        
        iterationSelect.addEventListener('change', (e) => {{
            stopPlay();
            currentIteration = parseInt(e.target.value);
            currentStep = 0;
            updateDisplay();
        }});
        
        speedSelect.addEventListener('change', () => {{
            if (isPlaying) {{
                stopPlay();
                startPlay();
            }}
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'ArrowRight':
                    e.preventDefault();
                    stopPlay();
                    nextStep();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stopPlay();
                    prevStep();
                    break;
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    stopPlay();
                    const currentIdx = iterations.indexOf(currentIteration);
                    if (currentIdx > 0) {{
                        currentIteration = iterations[currentIdx - 1];
                        currentStep = 0;
                        updateDisplay();
                    }}
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    stopPlay();
                    const idx = iterations.indexOf(currentIteration);
                    if (idx < iterations.length - 1) {{
                        currentIteration = iterations[idx + 1];
                        currentStep = 0;
                        updateDisplay();
                    }}
                    break;
            }}
        }});
        
        // Initialize display
        currentIteration = iterations[0];
        updateDisplay();
    </script>
</body>
</html>
"""
    
    # Save HTML file
    output_path = viz_path / 'viewer.html'
    output_path.write_text(html_content)
    print(f"✓ HTML viewer generated: {output_path}")
    print(f"\nTo view, open: {output_path}")
    print(f"Or run: open {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Find all test directories and generate viewers for each
        viz_base = Path(__file__).parent / 'aco_visualizations'
        if viz_base.exists():
            test_dirs = sorted([d for d in viz_base.iterdir() if d.is_dir() and d.name.startswith('test_')])
            if test_dirs:
                print(f"Found {len(test_dirs)} test directories")
                for viz_dir in test_dirs:
                    print(f"Generating viewer for: {viz_dir.name}")
                    generate_html_viewer(viz_dir)
                print(f"\n✓ Generated {len(test_dirs)} HTML viewers")
                print(f"\nTo view latest: open {test_dirs[-1] / 'viewer.html'}")
            else:
                print("No visualization directories found")
                sys.exit(1)
        else:
            print("Usage: python generate_html_viewer.py <path_to_visualization_directory>")
            sys.exit(1)
    else:
        viz_dir = Path(sys.argv[1])
        generate_html_viewer(viz_dir)
