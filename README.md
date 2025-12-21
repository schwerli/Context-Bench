# Context Retrieval Evaluation

## Installation

```bash
# Critical: Correct versions required
pip install "tree-sitter==0.20.4" tree-sitter-languages
```

## Usage

```bash
python -m contextbench_eval.evaluate \
    --gold <gold_path> \
    --pred <pred_path> \
    [--cache ./repos] \
    [--out results.jsonl]
```

## Example

```bash
cd /data/evaluation

# Use schwer environment (has tree-sitter installed)
/home/schwer/miniconda3/envs/schwer/bin/python3 -m contextbench_eval.evaluate \
    --gold Context-dataset/Verified/annots_pass \
    --pred traj_verified/psf__requests-1142/psf__requests-1142.traj.json \
    --out result.jsonl
```

## Metrics

**Three Granularities**:
- **File**: File path sets
- **Symbol**: Definition nodes (class/function/method) within viewed spans
- **Span**: Byte intervals

**Final Context**: Coverage & Precision  
**Trajectory**: Per-step coverage, AUC-Coverage, Redundancy  
**EditLoc**: Edit localization recall & precision

## Output

```json
{
  "instance_id": "...",
  "num_steps": 6,
  "final": {
    "file": {"coverage": 1.0, "precision": 0.5, ...},
    "symbol": {"coverage": 1.0, "precision": 0.125, ...},
    "span": {"coverage": 1.0, "precision": 0.126, ...}
  },
  "trajectory": {
    "steps": [{"step": 1, "coverage": {...}}, ...],
    "auc_coverage": {"file": 1.0, "symbol": 1.0, "span": 1.0},
    "redundancy": {"file": 0.5, "symbol": 0.58, "span": 0.12}
  },
  "editloc": {"recall": 0.782, "precision": 1.0, ...}
}
```

## Directory Structure

```
core/       # Intervals, fileio, repo checkout
parsers/    # Gold, trajectory, diff
extractors/ # Tree-sitter symbol extraction
metrics/    # Metric computation
evaluate.py # Main evaluation script
```
