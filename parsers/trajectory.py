"""Unified trajectory parsing interface."""

import json
import os
from typing import List, Tuple, Optional
from ..agents import extract_trajectory as extract_unified

class Step:
    """One retrieval step."""
    def __init__(self, files=None, spans=None, symbols=None):
        self.files = files or []
        self.spans = spans or []  # [{file, start_line, end_line}]
        self.symbols = symbols or {}  # {file: [symbolName, ...]}


def parse_trajectory(data: dict) -> Tuple[List[Step], Optional[Step]]:
    """Parse trajectory from unified agent data format.
    
    Args:
        data: dict with 'traj_data' containing:
            - pred_steps: list of {'files': [...], 'spans': {...}}
            - pred_files: final file list
            - pred_spans: final span dict
    
    Returns:
        (trajectory_steps, final_step)
    """
    traj_data = data.get('traj_data', {})
    
    # Convert pred_steps to Step objects
    traj_steps = []
    for step_data in traj_data.get('pred_steps', []):
        files = step_data.get('files', [])
        spans_dict = step_data.get('spans', {})
        symbols_dict = step_data.get('symbols', {}) or {}
        
        # Convert spans dict to list format
        spans = []
        for file_path, file_spans in spans_dict.items():
            for span in file_spans:
                spans.append({
                    'file': file_path,
                    'start_line': span['start'],
                    'end_line': span['end']
                })
        
        traj_steps.append(Step(files, spans, symbols_dict))
    
    # Build final step
    final_files = traj_data.get('pred_files', [])
    final_spans_dict = traj_data.get('pred_spans', {})
    final_symbols_dict = traj_data.get('pred_symbols', {}) or {}
    
    final_spans = []
    for file_path, file_spans in final_spans_dict.items():
        for span in file_spans:
            final_spans.append({
                'file': file_path,
                'start_line': span['start'],
                'end_line': span['end']
            })
    
    final_step = Step(final_files, final_spans, final_symbols_dict)
    
    # Fallback
    if not traj_steps:
        traj_steps = [final_step]
    
    return traj_steps, final_step


def load_traj_file(traj_file: str) -> dict:
    """Load trajectory file using unified agent interface."""
    result = extract_unified(traj_file)
    
    # Extract instance_id from filename
    basename = os.path.basename(traj_file)
    instance_id = ""
    if basename.endswith('.traj.json'):
        instance_id = basename.replace('.traj.json', '')
    elif basename.endswith('.checkpoints.jsonl'):
        instance_id = basename.replace('.checkpoints.jsonl', '')
    elif basename.endswith('_traj.json'):
        instance_id = basename.replace('_traj.json', '')
    else:
        instance_id = basename
    
    # Get model_patch if it's a .traj.json file
    model_patch = ""
    if traj_file.endswith('.traj.json'):
        with open(traj_file) as f:
            data = json.load(f)
            model_patch = data.get("info", {}).get("submission", "")
    elif traj_file.endswith('_traj.json'):
        with open(traj_file) as f:
            data = json.load(f)
            # Prefer explicit instance_id when present
            if isinstance(data, dict) and data.get("instance_id"):
                instance_id = data.get("instance_id")
            model_patch = data.get("6_final_selected_patch", "") if isinstance(data, dict) else ""
    
    return {
        "instance_id": instance_id,
        "traj_data": result,
        "model_patch": model_patch
    }


def load_pred(path: str) -> List[dict]:
    """Load prediction data from JSON/JSONL or trajectory files."""
    # Handle trajectory files directly (.traj.json or .checkpoints.jsonl)
    if path.endswith('.traj.json') or path.endswith('.checkpoints.jsonl') or path.endswith('_traj.json'):
        loaded = load_traj_file(path)
        return [loaded]
    
    # Handle regular JSON/JSONL prediction files
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        obj = json.load(f)
        if isinstance(obj, list):
            return obj
        return [obj]

