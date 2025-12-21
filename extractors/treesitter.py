"""Tree-sitter based definition extraction."""

import os
from typing import Dict, List, Optional, Set, Tuple

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
    """Check if tree-sitter is available."""
    return _TS_AVAILABLE

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
        return []
    
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
