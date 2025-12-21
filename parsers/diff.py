"""Parse unified diffs to extract edit locations."""

import os
import re
from typing import Dict, List, Tuple
from ..core import ByteInterval, merge
from ..core.fileio import line_to_byte

def parse_diff(diff_text: str, repo_dir: str) -> Dict[str, List[ByteInterval]]:
    """Extract edited byte ranges per file from unified diff."""
    edits = _parse_hunks(diff_text)
    result = {}
    
    for file_path, line_ranges in edits.items():
        abs_path = os.path.join(repo_dir, file_path)
        if not os.path.exists(abs_path):
            continue
        
        intervals = []
        for start, end, _ in line_ranges:
            span = line_to_byte(abs_path, start, end)
            if span:
                intervals.append(span)
        
        if intervals:
            result[file_path] = merge(intervals)
    
    return result

def _parse_hunks(diff_text: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """Parse diff hunks. Returns {file: [(start_line, end_line, change_type)]}."""
    result = {}
    current_file = None
    lines = diff_text.split('\n')
    
    hunk_re = re.compile(r'^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@')
    
    for i, line in enumerate(lines):
        if line.startswith('+++'):
            match = re.match(r'^\+\+\+\s+b/(.+)', line)
            if match:
                current_file = match.group(1)
        elif line.startswith('@@') and current_file:
            match = hunk_re.match(line)
            if match:
                new_start = int(match.group(1))
                # Parse hunk body
                j, new_line, edit_start, edit_type = i + 1, new_start, None, None
                while j < len(lines) and not lines[j].startswith(('@@', 'diff ', '---')):
                    if lines[j].startswith('+') and not lines[j].startswith('+++'):
                        if edit_start is None:
                            edit_start, edit_type = new_line, 'add'
                        new_line += 1
                    elif lines[j].startswith('-') and not lines[j].startswith('---'):
                        if edit_start is None:
                            edit_start, edit_type = new_line, 'del'
                    elif lines[j].startswith(' '):
                        if edit_start is not None:
                            result.setdefault(current_file, []).append(
                                (edit_start, new_line - 1, edit_type))
                            edit_start, edit_type = None, None
                        new_line += 1
                    j += 1
                if edit_start is not None:
                    result.setdefault(current_file, []).append(
                        (edit_start, new_line - 1, edit_type))
    return result

