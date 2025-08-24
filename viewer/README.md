# Trajectory Viewer

This directory contains a standalone HTML viewer for trajectory data that doesn't require a Python server.

## Files

- `trajectories.html` - Base HTML template
- `trajectories.js` - JavaScript functionality for viewing trajectories
- `trajectories.css` - Styling for the viewer
- `trajectories_standalone.html` - Generated standalone viewer (created by the script)

## Usage

### Method 1: Generate Standalone Viewer (Recommended)

1. Run the generation script using the main visualization tool:
   ```bash
   # From the project root directory
   python visualize_evals.py --task=zork --action=viewer
   ```

2. Open the generated HTML file directly in your browser:
   ```bash
   open viewer/trajectories_standalone.html
   ```

### Method 2: Custom JSON File

You can also specify a different JSON file:

```bash
python visualize_evals.py --task=zork --action=viewer --json_file=path/to/your/trajectories.json --output_html=viewer/custom_output.html
```

### Method 3: Manual Data Loading

If you want to load data programmatically, you can call:

```javascript
loadTrajectoryDataFromJSON(yourJsonData);
```

## Available Commands

The `visualize_evals.py` script supports multiple actions:

- `--action=eval`: Run evaluation analysis (default)
- `--action=render`: Export trajectory data as JSON
- `--action=viewer`: Generate standalone HTML viewer

Examples:
```bash
# Generate viewer for zork task
python visualize_evals.py --task=zork --action=viewer

# Generate viewer for connections task
python visualize_evals.py --task=connections --action=viewer

# Generate viewer with custom JSON file
python visualize_evals.py --task=zork --action=viewer --json_file=eval_results/custom/trajectories.json
```

## Features

- **No Server Required**: The generated HTML file contains all the data embedded directly
- **Standalone**: Just double-click the HTML file to open in any browser
- **Interactive Navigation**: Browse through trajectories with Previous/Next buttons
- **Statistics**: View total trajectories, completion status, and average steps
- **Formatted Display**: Clean presentation of conversation steps with role-based styling

## Data Format

The viewer expects JSON data in this format:

```json
[
  {
    "model": "model-name",
    "game": "game-name", 
    "status": "done|error",
    "conversation": [
      {
        "role": "user|assistant|system",
        "content": "message content",
        "reasoning": "optional reasoning"
      }
    ]
  }
]
```

## Troubleshooting

- **File not found**: Make sure the JSON file path is correct
- **Invalid JSON**: Check that your JSON file is properly formatted
- **Browser issues**: Try opening the HTML file in a different browser
- **Module not found**: Make sure you're running from the project root with the virtual environment activated
