"""
Sidebar UI components
"""

import sys
from typing import Any, Dict, List, Tuple

import streamlit as st

sys.path.append("/Users/colinwork/Documents/GitHub/docker_file_search/src")

# Import after path setup to avoid E402 errors
from utils.constants import (  # noqa: E402
    DEFAULT_FOLDER_PATH,
    DEFAULT_MAX_RESULTS,
    DEFAULT_READABILITY_RANGE,
    MAX_RESULTS_RANGE,
    READABILITY_RANGE,
    SEARCH_METHODS,
)


def render_configuration_section() -> Tuple[str, bool]:
    """Render the configuration section of the sidebar"""
    st.header("⚙️ Configuration")

    # Folder selection
    doc_folder = st.text_input(
        "Documents Folder Path",
        value=DEFAULT_FOLDER_PATH,
        help="Path to folder containing .docx files",
    )

    load_clicked = st.button("🔄 Load Documents", type="primary")

    return doc_folder, load_clicked


def render_filters_section(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Render the filters section of the sidebar"""
    if not documents:
        return {}

    st.header("🏷️ Filters")

    # Get available filter options
    categories = list(set([doc.get("business_category", "General") for doc in documents]))
    doc_types = list(set([doc.get("document_type", "Unknown") for doc in documents]))
    all_tags = list(set([tag for doc in documents for tag in doc.get("tags", [])]))

    # Category filter
    selected_category = st.selectbox(
        "Business Category", ["All"] + sorted(categories), key="category_filter"
    )

    # Document type filter
    selected_doc_type = st.selectbox(
        "Document Type", ["All"] + sorted(doc_types), key="doc_type_filter"
    )

    # Tags filter
    selected_tags = st.multiselect(
        "Tags", sorted(all_tags)[:20], key="tags_filter"  # Limit to top 20 tags
    )

    # Urgency filter
    selected_urgency = st.selectbox(
        "Urgency Level", ["All", "High", "Medium", "Low"], key="urgency_filter"
    )

    # Readability filter
    readability_range = st.slider(
        "Readability Score Range",
        min_value=READABILITY_RANGE[0],
        max_value=READABILITY_RANGE[1],
        value=DEFAULT_READABILITY_RANGE,
        key="readability_filter",
        help="Flesch Reading Ease score (higher = easier to read)",
    )

    return {
        "category": (selected_category if selected_category != "All" else None),
        "doc_type": (selected_doc_type if selected_doc_type != "All" else None),
        "tags": selected_tags,
        "urgency": (selected_urgency if selected_urgency != "All" else None),
        "readability_range": readability_range,
    }


def render_search_settings() -> Tuple[str, int]:
    """Render the search settings section of the sidebar"""
    st.header("🔍 Search Settings")

    search_method = st.selectbox(
        "Search Method", SEARCH_METHODS, help="Choose how to search through documents"
    )

    max_results = st.slider(
        "Max Results",
        min_value=MAX_RESULTS_RANGE[0],
        max_value=MAX_RESULTS_RANGE[1],
        value=DEFAULT_MAX_RESULTS,
    )

    return search_method, max_results


def render_sidebar(documents: List[Dict[str, Any]]) -> Tuple[str, bool, Dict[str, Any], str, int]:
    """Render the complete sidebar"""
    with st.sidebar:
        # Configuration section
        doc_folder, load_clicked = render_configuration_section()

        # Filters section
        filters = render_filters_section(documents)

        # Search settings
        search_method, max_results = render_search_settings()

        return doc_folder, load_clicked, filters, search_method, max_results
