"""
Utility functions for the document search application
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def highlight_query_terms(text: str, query_terms: List[str]) -> str:
    """Highlight query terms in text with markdown bold formatting"""
    highlighted = text
    for word in query_terms:
        highlighted = re.sub(f"({re.escape(word)})", r"**\1**", highlighted, flags=re.IGNORECASE)
    return highlighted


def format_tags_display(tags: List[str], max_tags: int = 10) -> str:
    """Format tags for display with a limit"""
    if not tags:
        return "No tags"
    return ", ".join(tags[:max_tags])


def format_file_size(size_kb: float) -> str:
    """Format file size in a human-readable format"""
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    elif size_kb < 1024 * 1024:
        return f"{size_kb / 1024:.1f} MB"
    else:
        return f"{size_kb / (1024 * 1024):.1f} GB"


def generate_filename_with_timestamp(base_name: str, extension: str) -> str:
    """Generate filename with current timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def safe_get_nested_value(dictionary: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value with default fallback"""
    try:
        for key in keys:
            dictionary = dictionary[key]
        return dictionary
    except (KeyError, TypeError):
        return default


def create_downloadable_summary(
    report_data: Dict[str, Any], format_type: str = "json"
) -> Optional[str]:
    """Create downloadable summary in different formats"""
    if format_type == "json":
        return json.dumps(report_data, indent=2, ensure_ascii=False)

    elif format_type == "csv":
        # Create CSV for document details
        if "document_details" in report_data:
            df = pd.DataFrame(report_data["document_details"])
            return df.to_csv(index=False)

    elif format_type == "text":
        # Create human-readable text summary
        metadata = report_data.get("metadata", {})
        stats = report_data.get("summary_statistics", {})

        text_summary = f"""
DOCUMENT SEARCH SUMMARY REPORT
Generated: {metadata.get('generated_date', 'Unknown')}

OVERVIEW
--------
Total Documents: {metadata.get('total_documents', 0)}
Search Query: {metadata.get('search_query') or 'N/A'}
Search Method: {metadata.get('search_method') or 'N/A'}

STATISTICS
----------
Total Word Count: {stats.get('total_word_count', 0):,}
Average Readability: {stats.get('average_readability', 0)}
Total Size: {stats.get('total_size_kb', 0)} KB

DOCUMENT CATEGORIES
-------------------
"""

        categories = stats.get("document_categories", {})
        for category, count in categories.items():
            text_summary += f"{category}: {count}\n"

        text_summary += "\nDOCUMENT TYPES\n--------------\n"
        doc_types = stats.get("document_types", {})
        for doc_type, count in doc_types.items():
            text_summary += f"{doc_type}: {count}\n"

        if "analytics" in report_data:
            text_summary += "\nTOP TAGS\n--------\n"
            top_tags = report_data["analytics"].get("top_tags", {})
            for tag, count in list(top_tags.items())[:10]:
                text_summary += f"{tag}: {count}\n"

        return text_summary

    return None


def validate_folder_path(folder_path: str) -> bool:
    """Validate if folder path exists and is accessible"""
    import os

    return os.path.exists(folder_path) and os.path.isdir(folder_path)


def extract_query_tags(query: str) -> List[str]:
    """Extract tags from a comma-separated query string"""
    return [tag.strip() for tag in query.split(",") if tag.strip()]
