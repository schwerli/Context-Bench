"""Definition extraction.

Primary path uses tree-sitter when available. If tree-sitter is unavailable in the
runtime environment, we fall back to a lightweight regex-based extractor for a
subset of languages. This keeps file/span evaluation usable and provides a
best-effort symbol metric.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Iterable

DefNode = Tuple[str, int, int]  # (kind, start_byte, end_byte)

# Language configuration
LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "c_sharp", ".php": "php",
    ".rb": "ruby", ".swift": "swift", ".kt": "kotlin", ".scala": "scala"
}

DEF_NODES = {
    "python": {"function_definition", "class_definition", "async_function_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "java": {"method_declaration", "class_declaration", "interface_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "ruby": {"method", "class", "module"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
    "scala": {"function_definition", "class_definition", "trait_definition"}
}

_TS_AVAILABLE = False
_PARSERS = {}

# Try new API first (tree-sitter >= 0.21)
try:
    import tree_sitter as ts
    from tree_sitter_languages import get_language
    _TS_AVAILABLE = True
    _API_TYPE = "new"
except ImportError:
    try:
        # Fallback to old API
        from tree_sitter_languages import get_parser as _get_parser
        _TS_AVAILABLE = True
        _API_TYPE = "old"
    except ImportError:
        _TS_AVAILABLE = False
        _API_TYPE = None

def available() -> bool:
    """Check if definition extraction is available."""
    return True

def _get_parser_for_lang(lang: str):
    """Get parser for language (handles both API versions)."""
    if not _TS_AVAILABLE:
        return None
    
    if lang in _PARSERS:
        return _PARSERS[lang]
    
    try:
        # Always use get_parser from tree_sitter_languages
        from tree_sitter_languages import get_parser as _get_parser
        parser = _get_parser(lang)
        _PARSERS[lang] = parser
        return parser
    except Exception as e:
        _PARSERS[lang] = None
        return None

def extract_defs(file_path: str) -> List[DefNode]:
    """Extract definition nodes from file."""
    if not _TS_AVAILABLE:
        return [(k, s, e) for k, _n, s, e in extract_named_defs(file_path)]
    
    lang = LANG_MAP.get(os.path.splitext(file_path.lower())[1])
    if not lang or lang not in DEF_NODES:
        return []
    
    parser = _get_parser_for_lang(lang)
    if not parser:
        return []
    
    try:
        with open(file_path, 'rb') as f:
            tree = parser.parse(f.read())
    except Exception:
        return []
    
    result = []
    target_types = DEF_NODES[lang]
    exclude_from_result = {"program", "module", "source_file", "translation_unit"}
    
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if not getattr(node, "is_named", False):
            continue
        
        node_type = getattr(node, "type", "")
        if "comment" in node_type:
            continue
        
        # Add to result if it's a definition (but still traverse children)
        if node_type in target_types:
            result.append((node_type, node.start_byte, node.end_byte))
        
        # Always traverse children (don't skip based on exclude list)
        for child in reversed(getattr(node, "children", [])):
            stack.append(child)
    
    return result

def _node_text(src: bytes, node) -> str:
    try:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


_IDENTIFIER_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "scoped_identifier",
    "namespace_identifier",
    "method_identifier",
    "constant_identifier",
}


def _iter_descendants(node) -> Iterable:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in reversed(getattr(cur, "children", []) or []):
            stack.append(child)


def _best_name_for_def(def_node, src: bytes) -> str:
    """Best-effort extraction of a definition name for a tree-sitter def node."""
    # Many languages expose a name field directly.
    try:
        name_node = def_node.child_by_field_name("name")
    except Exception:
        name_node = None

    if name_node is not None:
        name = _node_text(src, name_node).strip()
        if name:
            return name

    # C/C++ often expose declarator.
    try:
        decl = def_node.child_by_field_name("declarator")
    except Exception:
        decl = None

    search_root = decl if decl is not None else def_node

    # Heuristic: prefer plain identifiers over type identifiers.
    best = ""
    best_rank = 10**9
    rank_map = {"identifier": 0, "field_identifier": 1, "property_identifier": 2, "method_identifier": 3}
    for n in _iter_descendants(search_root):
        if not getattr(n, "is_named", False):
            continue
        t = getattr(n, "type", "")
        if t not in _IDENTIFIER_TYPES:
            continue
        txt = _node_text(src, n).strip()
        if not txt:
            continue
        r = rank_map.get(t, 100)
        if r < best_rank:
            best = txt
            best_rank = r
            if best_rank == 0:
                break
    return best


def extract_named_defs(file_path: str) -> List[Tuple[str, str, int, int]]:
    """Extract named definitions from file.

    Returns [(kind, name, start_byte, end_byte)].
    """
    if not _TS_AVAILABLE:
        return _extract_named_defs_fallback(file_path)

    lang = LANG_MAP.get(os.path.splitext(file_path.lower())[1])
    if not lang or lang not in DEF_NODES:
        return []

    parser = _get_parser_for_lang(lang)
    if not parser:
        return []

    try:
        with open(file_path, "rb") as f:
            src = f.read()
        tree = parser.parse(src)
    except Exception:
        return []

    result: List[Tuple[str, str, int, int]] = []
    target_types = DEF_NODES[lang]

    for node in _iter_descendants(tree.root_node):
        if not getattr(node, "is_named", False):
            continue
        node_type = getattr(node, "type", "")
        if "comment" in node_type:
            continue
        if node_type not in target_types:
            continue

        name = _best_name_for_def(node, src)
        if not name:
            continue
        result.append((node_type, name, node.start_byte, node.end_byte))

    return result


def extract_def_set_from_symbol_names(
    pred_symbols_by_file: Dict[str, List[str]],
    repo_dir: str,
) -> Set[Tuple[str, str, int, int]]:
    """Map predicted symbol names to tree-sitter def byte ranges.

    Returns {(file, kind, start_byte, end_byte)}.
    """
    out: Set[Tuple[str, str, int, int]] = set()
    if not pred_symbols_by_file:
        return out

    for rel_path, symbols in pred_symbols_by_file.items():
        if not rel_path:
            continue
        if not isinstance(symbols, list) or not symbols:
            continue
        abs_path = os.path.join(repo_dir, rel_path)
        if not os.path.exists(abs_path):
            continue

        named_defs = extract_named_defs(abs_path)
        if not named_defs:
            continue

        by_name: Dict[str, List[Tuple[str, int, int]]] = {}
        for kind, name, s, e in named_defs:
            by_name.setdefault(name, []).append((kind, s, e))

        for raw in symbols:
            if not raw or not isinstance(raw, str):
                continue
            sym = raw.strip()
            if not sym:
                continue
            candidates = [sym]
            if "." in sym:
                candidates.append(sym.split(".")[-1])

            matched = False
            for cand in candidates:
                defs = by_name.get(cand)
                if not defs:
                    continue
                for kind, s, e in defs:
                    out.add((rel_path, kind, s, e))
                matched = True
                break

            if not matched:
                continue

    return out


def extract_def_set_in_spans(spans_by_file: Dict[str, List[Tuple[int, int]]], repo_dir: str) -> Set[Tuple[str, str, int, int]]:
    """
    Extract definitions that overlap with given byte spans.
    Returns {(file, kind, start_byte, end_byte)}.
    """
    result = set()
    for file_path, byte_intervals in spans_by_file.items():
        abs_path = os.path.join(repo_dir, file_path)
        if not os.path.exists(abs_path):
            continue
        
        # Get all definitions in this file
        all_defs = extract_defs(abs_path)
        
        # Keep only definitions that overlap with our spans
        for kind, def_start, def_end in all_defs:
            # Check if this definition overlaps any of our spans
            for span_start, span_end in byte_intervals:
                # Overlap check: def and span have any byte in common
                if not (def_end < span_start or def_start > span_end):
                    result.add((file_path, kind, def_start, def_end))
                    break  # Already added, no need to check other spans
    
    return result


def _extract_named_defs_fallback(file_path: str) -> List[Tuple[str, str, int, int]]:
    """Regex-based definition extraction fallback.

    This is intentionally conservative and only supports a subset of languages.
    The goal is stable byte ranges for overlap checks, not perfect parsing.
    """
    ext = os.path.splitext(file_path.lower())[1]
    try:
        with open(file_path, "rb") as f:
            src = f.read()
    except OSError:
        return []

    if not src:
        return []

    patterns: List[Tuple[str, "re.Pattern[bytes]", int]] = []
    import re

    if ext == ".py":
        patterns = [
            ("function", re.compile(rb"^[ \t]*(?:async[ \t]+def|def)[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.M), 1),
            ("class", re.compile(rb"^[ \t]*class[ \t]+([A-Za-z_]\w*)[ \t]*(?:\(|:)", re.M), 1),
        ]
    elif ext in {".c", ".h", ".cc", ".cpp", ".hpp"}:
        # Function signature with opening brace on the same line (common style).
        patterns = [
            ("function", re.compile(rb"^[ \t]*(?:[A-Za-z_]\w*[ \t]+)+\**[ \t]*([A-Za-z_]\w*)[ \t]*\([^;{]*\)\s*\{", re.M), 1),
            ("struct", re.compile(rb"^[ \t]*struct[ \t]+([A-Za-z_]\w*)[ \t]*\{", re.M), 1),
        ]
    else:
        return []

    hits: List[Tuple[int, int, str, str]] = []  # (start, match_end, kind, name)
    for kind, rx, name_group in patterns:
        for m in rx.finditer(src):
            try:
                name = m.group(name_group).decode("utf-8", errors="replace").strip()
            except Exception:
                name = ""
            if not name:
                continue
            hits.append((m.start(), m.end(), kind, name))

    if not hits:
        return []

    hits.sort(key=lambda x: x[0])
    out: List[Tuple[str, str, int, int]] = []
    for i, (s, _me, kind, name) in enumerate(hits):
        if i + 1 < len(hits):
            e = max(s, hits[i + 1][0] - 1)
        else:
            e = max(s, len(src) - 1)
        out.append((kind, name, s, e))
    return out
