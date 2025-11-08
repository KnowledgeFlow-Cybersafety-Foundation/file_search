import re
from typing import Dict, List, Tuple


def find_matches(docs: Dict[str, str], term: str) -> List[Tuple[str, str]]:
    """Return list of (document_name, snippet) for occurrences of a search term."""
    term_lower = term.lower()
    results = []
    pattern = re.compile(rf"(.{{0,120}}\b{re.escape(term_lower)}\b.*?\. )", re.IGNORECASE)
    for name, content in docs.items():
        if term_lower in content.lower():
            match = pattern.search(content.replace("\n", " "))
            snippet = (
                match.group(0).strip()
                if match
                else content[:160] + ("..." if len(content) > 160 else "")
            )
            results.append((name, snippet))
    return results
