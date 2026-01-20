"""Unified agent trajectory extraction interface."""

from .minisweagent import extract_trajectory as extract_miniswe
from .sweagent import extract_trajectory as extract_swe
from .agentless import extract_trajectory as extract_agentless


def extract_trajectory(traj_file: str) -> dict:
    """Auto-detect format and extract trajectory.
    
    Supports:
    - MiniSWE-agent: .traj.json files
    - SWE-agent: .checkpoints.jsonl files
    - Agentless: *_traj.json files
    
    Returns unified format:
    {
        'pred_steps': [{'files': [...], 'spans': {...}}, ...],
        'pred_files': [...],
        'pred_spans': {...}
    }
    """
    if traj_file.endswith('.checkpoints.jsonl'):
        return extract_swe(traj_file)
    elif traj_file.endswith('.traj.json'):
        return extract_miniswe(traj_file)
    elif traj_file.endswith('_traj.json'):
        return extract_agentless(traj_file)
    else:
        raise ValueError(f"Unsupported trajectory format: {traj_file}")


__all__ = ['extract_trajectory']
