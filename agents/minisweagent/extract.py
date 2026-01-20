"""Extract trajectory from MiniSWE-agent format."""

import json
import os
import re
from typing import Dict, List, Any


def extract_trajectory(traj_file: str) -> Dict[str, Any]:
    """Extract trajectory steps and final context from MiniSWE-agent .traj.json file.
    
    Args:
        traj_file: Path to .traj.json file
        
    Returns:
        Dictionary with:
        - pred_steps: List of trajectory steps with files and spans
        - pred_files: Final context files
        - pred_spans: Final context spans
    """
    with open(traj_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    steps = []
    
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        match = re.search(r"```bash\s*\n(.*?)\n```", msg.get("content", ""), re.DOTALL)
        if not match:
            continue
        
        cmd = match.group(1).strip()
        if "COMPLETE_TASK" in cmd:
            continue
        
        views = _extract_views_from_command(cmd)
        if views:
            step_data = {
                'files': [],
                'spans': {}
            }
            
            for view in views:
                file_path = view['file']
                step_data['files'].append(file_path)
                
                if 'start_line' in view and 'end_line' in view:
                    if file_path not in step_data['spans']:
                        step_data['spans'][file_path] = []
                    step_data['spans'][file_path].append({
                        'type': 'line',
                        'start': view['start_line'],
                        'end': view['end_line']
                    })
            
            if step_data['files']:
                steps.append(step_data)
    
    final_context = None
    patch_ctx = data.get("info", {}).get("patch_context_data", {}).get("patch_context", "")
    if patch_ctx:
        parsed = _parse_patch_context(patch_ctx)
        if parsed:
            final_context = {
                'files': list(parsed.keys()),
                'spans': parsed
            }
    
    result = {'pred_steps': steps}
    
    if final_context:
        result['pred_files'] = final_context['files']
        result['pred_spans'] = final_context['spans']
    elif steps:
        all_files = set()
        all_spans = {}
        for step in steps:
            all_files.update(step['files'])
            for file_path, spans in step['spans'].items():
                if file_path not in all_spans:
                    all_spans[file_path] = []
                all_spans[file_path].extend(spans)
        result['pred_files'] = sorted(all_files)
        result['pred_spans'] = all_spans
    else:
        result['pred_files'] = []
        result['pred_spans'] = {}
    
    return result


def _extract_views_from_command(cmd: str) -> List[Dict]:
    """Extract file viewing operations from command."""
    if any(p in cmd for p in ['sed -i', 'echo ', 'mkdir', 'rm ', 'git add', 'git commit']):
        return []
    
    # sed -n 'start,endp' file
    m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s&|>;<]+)", cmd)
    if m:
        f = m.group(3).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source_file(f):
            return [{'file': f, 'start_line': int(m.group(1)), 'end_line': int(m.group(2))}]
    
    # nl file | sed -n 'start,endp'
    m = re.search(r"nl\s+[^\|]+\s+([^\s\|]+)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source_file(f):
            return [{'file': f, 'start_line': int(m.group(2)), 'end_line': int(m.group(3))}]
    
    # cat file
    m = re.search(r"\bcat\s+([^\s&|>]+)", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source_file(f):
            return [{'file': f}]
    
    # head -n N file
    m = re.search(r"\bhead\s+-n\s+(\d+)\s+([^\s&|>]+)", cmd)
    if m:
        f = m.group(2).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        if _is_source_file(f):
            return [{'file': f, 'start_line': 1, 'end_line': int(m.group(1))}]
    
    # grep file
    m = re.search(r"\bgrep\s+.*?\s+([^\s&|>]+\.(?:py|js|java|go|rs|c|cpp|h|hpp|ts|tsx))\b", cmd)
    if m:
        f = m.group(1).strip("'\"")
        if f.startswith('/testbed/'):
            f = f[9:]
        return [{'file': f}]
    
    return []


def _is_source_file(path: str) -> bool:
    """Check if path looks like source file."""
    exts = ['.py', '.js', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
            '.ts', '.tsx', '.jsx', '.rb', '.php', '.cs', '.kt', '.scala', '.swift']
    return any(path.endswith(e) for e in exts) or '/' in path


def _parse_patch_context(text: str) -> Dict[str, List[Dict]]:
    """Parse patch_context string format (File: path\\nLines: start-end)."""
    result = {}
    current_file = None
    
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
                if current_file not in result:
                    result[current_file] = []
                result[current_file].append({
                    'type': 'line',
                    'start': int(m.group(1)),
                    'end': int(m.group(2))
                })
    
    return result

