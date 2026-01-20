import json
import re
from typing import List, Dict, Any, Optional, Tuple


def extract_view_command(action: str) -> Optional[Tuple[str, int, int]]:
    """Extract file path and line range from str_replace_editor view command.
    
    Returns (file_path, start_line, end_line) or None if not a view command.
    """
    if not action or 'str_replace_editor view' not in action:
        return None
    
    match = re.search(r'str_replace_editor view\s+(\S+)(?:\s+--view_range\s+(\d+)\s+(\d+))?', action)
    if not match:
        return None
    
    file_path = match.group(1)
    if match.group(2) and match.group(3):
        start_line = int(match.group(2))
        end_line = int(match.group(3))
        return (file_path, start_line, end_line)
    else:
        return (file_path, 1, -1)


def extract_content_from_observation(observation: str) -> str:
    """Extract actual content from observation, removing headers like 'Here's'."""
    if not observation:
        return ""
    
    lines = observation.split('\n')
    result_lines = []
    skip_header = False
    
    for i, line in enumerate(lines):
        lower_line = line.lower().strip()
        if i == 0 and (lower_line.startswith("here's") or lower_line.startswith("here is")):
            skip_header = True
            continue
        if skip_header and i == 1:
            skip_header = False
            continue
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()


def _normalize_file_path(file_path: str) -> str:
    """Normalize file path by removing common prefixes."""
    # Remove /testbed/ prefix
    if file_path.startswith('/testbed/'):
        file_path = file_path[9:]
    # Remove leading /
    elif file_path.startswith('/'):
        file_path = file_path[1:]
    return file_path


def parse_patch_context(patch_context_str: str) -> Dict[str, Any]:
    """Parse patch_context string format.
    
    Format:
    File: /path/to/file
    Lines: start-end
    File: /another/file
    Lines: start-end
    """
    files = {}
    lines = patch_context_str.strip().split('\n')
    current_file = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('File:'):
            file_path = line.replace('File:', '').strip()
            current_file = _normalize_file_path(file_path)
            if current_file not in files:
                files[current_file] = []
        elif line.startswith('Lines:') and current_file:
            range_str = line.replace('Lines:', '').strip()
            if '-' in range_str:
                start, end = range_str.split('-')
                files[current_file].append({
                    'type': 'line',
                    'start': int(start),
                    'end': int(end)
                })
    
    return files


def extract_trajectory(checkpoint_file: str) -> Dict[str, Any]:
    """Extract trajectory steps and final context from SWE-agent checkpoint file.
    
    Args:
        checkpoint_file: Path to .checkpoints.jsonl file
        
    Returns:
        Dictionary with:
        - pred_steps: List of trajectory steps with files and spans
        - pred_files: Final context files (if patch_context exists)
        - pred_spans: Final context spans (if patch_context exists)
    """
    steps = []
    final_context = None
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            checkpoint = json.loads(line)
            
            if checkpoint.get('type') == 'patch_context':
                patch_context_str = checkpoint.get('patch_context', '')
                if patch_context_str:
                    parsed = parse_patch_context(patch_context_str)
                    final_context = {
                        'files': list(parsed.keys()),
                        'spans': parsed
                    }
                continue
            
            action = checkpoint.get('action', '')
            observation = checkpoint.get('observation', '')
            
            view_result = extract_view_command(action)
            if view_result:
                file_path, start_line, end_line = view_result
                
                if start_line > 0 and end_line > 0:
                    content = extract_content_from_observation(observation)
                    
                    # Normalize file path
                    normalized_path = _normalize_file_path(file_path)
                    
                    step_data = {
                        'files': [normalized_path],
                        'spans': {
                            normalized_path: [
                                {'type': 'line', 'start': start_line, 'end': end_line}
                            ]
                        }
                    }
                    
                    steps.append(step_data)
    
    result = {'pred_steps': steps}
    
    if final_context:
        result['pred_files'] = final_context['files']
        result['pred_spans'] = final_context['spans']
    elif steps:
        last_step = steps[-1]
        result['pred_files'] = last_step.get('files', [])
        result['pred_spans'] = last_step.get('spans', {})
    else:
        result['pred_files'] = []
        result['pred_spans'] = {}
    
    return result

