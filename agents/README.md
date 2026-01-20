# Agent Trajectory Extractors

This module provides unified trajectory extraction for different coding agents.

## Supported Agents

### 1. MiniSWE-agent
- **Format**: `.traj.json` files
- **Location**: `minisweagent/extract.py`
- **Features**:
  - Extracts file views from bash commands in messages
  - Supports `cat`, `sed -n`, `head`, `grep`, `nl | sed` commands
  - Parses `patch_context_data.patch_context` for final context
  - Returns model patch from `info.submission`

### 2. SWE-agent
- **Format**: `.checkpoints.jsonl` files  
- **Location**: `sweagent/extract.py`
- **Features**:
  - Extracts from `str_replace_editor view` commands with `--view_range`
  - Only includes steps with explicit line ranges
  - Parses `patch_context` string format (File:/Lines:)
  - No model patch extraction (not in checkpoints format)

## Unified Interface

```python
from contextbench_eval.agents import extract_trajectory

# Auto-detects format based on file extension
result = extract_trajectory('path/to/trajectory.traj.json')  # MiniSWE
result = extract_trajectory('path/to/trajectory.checkpoints.jsonl')  # SWE

# Returns unified format:
{
    'pred_steps': [
        {
            'files': ['file1.py', 'file2.py'],
            'spans': {
                'file1.py': [
                    {'type': 'line', 'start': 10, 'end': 20}
                ]
            }
        },
        ...
    ],
    'pred_files': ['file1.py', 'file2.py'],
    'pred_spans': {
        'file1.py': [
            {'type': 'line', 'start': 10, 'end': 50}
        ]
    }
}
```

## Adding New Agents

1. Create a new directory: `agents/newagent/`
2. Implement `extract.py` with `extract_trajectory(traj_file: str) -> dict`
3. Return the unified format shown above
4. Update `agents/__init__.py` to register the new agent
5. Add file extension detection logic

Example structure:
```
agents/
├── __init__.py          # Unified interface with auto-detection
├── minisweagent/
│   ├── __init__.py
│   └── extract.py       # MiniSWE-specific extraction
├── sweagent/
│   ├── __init__.py
│   └── extract.py       # SWE-specific extraction
└── newagent/
    ├── __init__.py
    └── extract.py       # Your agent's extraction logic
```

## Testing

```bash
# Test MiniSWE format
python -m contextbench_eval.evaluate \
    --gold Context-dataset/Verified/annots_pass \
    --pred traj_verified-mini/instance/instance.traj.json \
    --out results.jsonl

# Test SWE format  
python -m contextbench_eval.evaluate \
    --gold Context-dataset/Verified/annots_pass \
    --pred traj_verified-swe/instance/instance.checkpoints.jsonl \
    --out results.jsonl
```

## Notes

- All file paths should be relative (remove `/testbed/` prefix)
- Line numbers are 1-indexed
- Spans use inclusive ranges: `[start, end]`
- Empty steps (no spans) are filtered out automatically
- Final context is used if no valid steps are found

