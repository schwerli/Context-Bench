"""Git repository checkout management."""

import os
import re
import subprocess
import sys
from typing import Optional

def checkout(repo_url: str, commit: str, cache_dir: str, verbose: bool = True) -> Optional[str]:
    """Checkout repo at specific commit. Direct clone, no mirror."""
    if not repo_url or not commit or not cache_dir:
        return None
    
    repo_key = _normalize_url(repo_url)
    target_dir = os.path.join(cache_dir, repo_key)
    
    # Check if already exists and at correct commit
    if os.path.isdir(target_dir):
        if _verify_commit(target_dir, commit):
            if verbose:
                print(f"  Using existing checkout", file=sys.stderr)
            return target_dir
        else:
            # Wrong commit, checkout the correct one
            if verbose:
                print(f"  Checking out commit {commit[:12]}", file=sys.stderr)
            _git(["checkout", "-q", commit], cwd=target_dir)
            if _verify_commit(target_dir, commit):
                return target_dir
    
    # Clone repo
    if verbose:
        print(f"  Cloning {repo_url}", file=sys.stderr)
    
    os.makedirs(cache_dir, exist_ok=True)
    result = _git(["clone", "--progress", repo_url, target_dir], show_progress=True)
    
    if result.returncode != 0 or not os.path.isdir(target_dir):
        if verbose:
            print(f"  Clone failed", file=sys.stderr)
        return None
    
    # Checkout specific commit
    if verbose:
        print(f"  Checking out commit {commit[:12]}", file=sys.stderr)
    
    _git(["checkout", "-q", commit], cwd=target_dir)
    
    if _verify_commit(target_dir, commit):
        return target_dir
    
    return None

def _normalize_url(url: str) -> str:
    """Convert git URL to directory-safe name."""
    s = re.sub(r"^https?://", "", url.strip())
    s = re.sub(r"^git@", "", s).replace(":", "/").rstrip("/")
    s = s.replace("/", "__").replace(".git", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or "repo"

def _git(args, cwd=None, show_progress=False):
    """Run git command."""
    if show_progress:
        # Show real-time output for clone/fetch operations
        return subprocess.run(
            ["git"] + args, cwd=cwd, check=False, timeout=600
        )
    else:
        return subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True,
            text=True, check=False, timeout=600
        )

def _verify_commit(work_dir: str, expected: str) -> bool:
    """Check if working directory is at expected commit."""
    result = _git(["rev-parse", "HEAD"], cwd=work_dir)
    return result.returncode == 0 and result.stdout.strip() == expected
