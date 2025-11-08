"""
Main Streamlit application for Document Search Dashboard
"""

import streamlit as st
import os
import sys

# Add src directory to path for imports
sys.path.append("/Users/colinwork/Documents/GitHub/docker_file_search/src")

# Import modular components
from core.document_searcher import DocumentSearcher
from ui.sidebar import render_sidebar
from ui.tabs import (
    render_overview_tab,
    render_search_tab,
    render_tag_cloud_tab,
    render_analytics_tab,
    render_summary_report_tab,
)
from ui.components import display_document_metrics
from utils.constants import PAGE_CONFIG
from utils.helpers import validate_folder_path

# Page configuration
st.set_page_config(**PAGE_CONFIG)


# Initialize the searcher
@st.cache_resource
def get_searcher():
    return DocumentSearcher()


def display_sample_structure():
    """Display sample folder structure for users"""
    st.info("👈 Please load documents using the sidebar to get started!")

    st.markdown("### Expected Folder Structure")
    st.code(
        """
documents/
├── document1.docx
├── document2.docx
├── document3.docx
└── ...
    """
    )


def main():
    """Main application function"""
    st.title("📄 Document Search Dashboard")
    st.markdown("Search through your Word documents with ease!")

    searcher = get_searcher()

    # Render sidebar and get user inputs
    doc_folder, load_clicked, filters, search_method, max_results = render_sidebar(
        searcher.documents
    )

    # Handle document loading
    if load_clicked:
        if validate_folder_path(doc_folder):
            with st.spinner("Loading documents..."):
                searcher.load_documents(doc_folder)
            st.rerun()
        else:
            st.error("Folder not found!")

    # Main content area
    if not searcher.documents:
        display_sample_structure()
        return

    # Display initial document statistics
    display_document_metrics(searcher.documents)

    # Apply filters to get filtered documents
    filtered_docs = searcher.documents.copy()
    if filters:
        readability_range = filters.get("readability_range", [0, 100])
        filtered_docs = searcher.filter_documents(
            category=filters.get("category"),
            doc_type=filters.get("doc_type"),
            tags=filters.get("tags"),
            urgency=filters.get("urgency"),
            min_readability=readability_range[0],
            max_readability=readability_range[1],
        )

        # Update searcher's documents temporarily for search
        original_docs = searcher.documents
        searcher.documents = filtered_docs
        searcher._build_search_index()

    # Main tabs
    if searcher.documents:
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Overview", "🔍 Search", "☁️ Tag Cloud Search", "📈 Analytics", "📄 Summary Report"]
        )

        with tab1:
            render_overview_tab(filtered_docs, searcher)

        with tab2:
            render_search_tab(filtered_docs, searcher, search_method, max_results)

        with tab3:
            render_tag_cloud_tab(filtered_docs, searcher, max_results)

        with tab4:
            render_analytics_tab(filtered_docs)

        with tab5:
            render_summary_report_tab(filtered_docs, search_method)

        # Show filtered document count
        if "original_docs" in locals():
            if len(filtered_docs) < len(original_docs):
                st.info(
                    f"Showing {len(filtered_docs)} of " f"{len(original_docs)} documents (filtered)"
                )

    # Restore original documents
    if "original_docs" in locals():
        searcher.documents = original_docs
        searcher._build_search_index()


if __name__ == "__main__":
    main()
