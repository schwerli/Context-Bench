"""Extract trajectory steps from agent logs."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

class Step:
    """One retrieval step."""
    def __init__(self, files=None, spans=None):
        self.files = files or []
        self.spans = spans or []  # [{file, start_line, end_line}]

def parse_trajectory(data: dict) -> Tuple[List[Step], Optional[Step]]:
    """
    Extract (trajectory_steps, final_context) from traj data.
    trajectory_steps: incremental file views from messages
    final_context: LLM-filtered context from patch_context_data
    """
    # Parse messages for trajectory
    traj_steps = []
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        match = re.search(r"```bash\s*\n(.*?)\n```", msg.get("content", ""), re.DOTALL)
        if not match:
            continue
        
        cmd = match.group(1).strip()
        if "COMPLETE_TASK" in cmd:
            continue
        
        views = _extract_views(cmd)
        if views:
            files = [v['file'] for v in views]
            spans = [v for v in views if 'start_line' in v]
            traj_steps.append(Step(files, spans))
    
    # Parse final filtered context
    final_step = None
    patch_ctx = data.get("patch_context_data", {}).get("patch_context", "")
    if patch_ctx:
        spans = _parse_patch_context(patch_ctx)
        if spans:
            files = sorted(set(s['file'] for s in spans))
            final_step = Step(files, spans)
    
    # Fallback
    if not final_step:
        if traj_steps:
            all_files, all_spans = set(), []
            for s in traj_steps:
                all_files.update(s.files)
                all_spans.extend(s.spans)
            final_step = Step(sorted(all_files), all_spans)
        else:
            final_step = Step()
    
    if not traj_steps:
        traj_steps = [final_step]
    
    return traj_steps, final_step

def _extract_views(cmd: str) -> List[Dict]:
    """Extract file viewing operations from command."""
    if any(p in cmd for p in ['sed -i', 'echo ', 'mkdir', 'rm ', 'git add', 'git commit']):
        return []
    
    # sed -n 'start,endp' file
    m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s&|>;<]+)", cmd)
    if m:
        f = m.group(3).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source(f):
            return [{'file': f, 'start_line': int(m.group(1)), 'end_line': int(m.group(2))}]
    
    # nl file | sed -n 'start,endp'
    m = re.search(r"nl\s+[^\|]+\s+([^\s\|]+)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source(f):
            return [{'file': f, 'start_line': int(m.group(2)), 'end_line': int(m.group(3))}]
    
    # cat file
    m = re.search(r"\bcat\s+([^\s&|>]+)", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source(f):
            return [{'file': f}]
    
    # head -n N file
    m = re.search(r"\bhead\s+-n\s+(\d+)\s+([^\s&|>]+)", cmd)
    if m:
        f = m.group(2).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source(f):
            return [{'file': f, 'start_line': 1, 'end_line': int(m.group(1))}]
    
    # grep file
    m = re.search(r"\bgrep\s+.*?\s+([^\s&|>]+\.(?:py|js|java|go|rs|c|cpp|h|hpp|ts|tsx))\b", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        return [{'file': f}]
    
    return []

def _is_source(path: str) -> bool:
    """Check if path looks like source file."""
    exts = ['.py', '.js', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
            '.ts', '.tsx', '.jsx', '.rb', '.php', '.cs', '.kt', '.scala', '.swift']
    return any(path.endswith(e) for e in exts) or '/' in path

def _parse_patch_context(text: str) -> List[Dict]:
    """Parse LLM-filtered patch context (File: path\\nLines: start-end)."""
    spans, current_file = [], None
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('File:'):
            f = line[5:].strip()
            if f.startswith('/testbed/'):
                f = f[9:]
            elif f.startswith('/'):
                f = f[1:]
            current_file = f
        elif line.startswith('Lines:') and current_file:
            m = re.match(r'(\d+)-(\d+)', line[6:].strip())
            if m:
                spans.append({
                    'file': current_file,
                    'start_line': int(m.group(1)),
                    'end_line': int(m.group(2))
                })
    return spans

def load_pred(path: str) -> List[dict]:
    """Load prediction data from JSON/JSONL."""
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        obj = json.load(f)
        if isinstance(obj, list):
            return obj
        # Handle .traj.json format
        if "info" in obj and "messages" in obj:
            basename = os.path.basename(path)
            instance_id = basename.replace(".traj.json", "") if basename.endswith(".traj.json") else None
            return [{
                "instance_id": instance_id,
                "messages": obj.get("messages", []),
                "model_patch": obj.get("info", {}).get("submission", ""),
                "patch_context_data": obj.get("info", {}).get("patch_context_data", {})
            }]
        return [obj]

