"""
Tab components for the main application
"""

import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append("/Users/colinwork/Documents/GitHub/docker_file_search/src")

# Import after path setup to avoid E402 errors
from core.report_generator import ReportGenerator  # noqa: E402
from ui.components import (  # noqa: E402
    display_analytics_charts,
    display_category_charts,
    display_document_metrics,
    display_entities,
    display_search_result,
    display_tags_chart,
)
from utils.constants import MIME_TYPES, REPORT_FORMATS  # noqa: E402
from utils.helpers import (  # noqa: E402
    create_downloadable_summary,
    generate_filename_with_timestamp,
)


def render_overview_tab(documents: List[Dict[str, Any]], searcher) -> None:
    """Render the overview tab"""
    st.header("📊 Document Overview")

    if not documents:
        st.info("No documents to display")
        return

    # Display metrics
    display_document_metrics(documents)

    # Category and Type Distribution
    display_category_charts(documents)

    # Tags display
    display_tags_chart(documents)


def render_search_tab(
    documents: List[Dict[str, Any]], searcher, search_method: str, max_results: int
) -> None:
    """Render the search tab"""
    # Search interface
    search_query = st.text_input(
        "🔍 Search Query",
        placeholder="Enter keywords to search for...",
        help="Search through filtered documents",
    )

    if search_query:
        with st.spinner("Searching..."):
            if search_method == "tags":
                query_tags = [tag.strip() for tag in search_query.split(",")]
                results = searcher.search_by_tags(query_tags, max_results)
            else:
                results = searcher.search(search_query, search_method, max_results)

        if results:
            st.success(f"Found {len(results)} results for '{search_query}'")

            # Display results
            for i, result in enumerate(results, 1):
                display_search_result(result, i, search_query, search_method)
        else:
            st.warning(f"No results found for '{search_query}'")


def render_tag_cloud_tab(documents: List[Dict[str, Any]], searcher, max_results: int) -> None:
    """Render the tag cloud search tab"""
    st.header("☁️ Tag Cloud Search")
    st.markdown("Click on tags in the word cloud or select from the list " "to search documents!")

    # Generate word cloud for tags
    wordcloud_data = searcher.generate_wordcloud_tags(documents)

    if wordcloud_data:
        wordcloud, tag_freq = wordcloud_data

        # Display word cloud
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

        # Interactive tag selection
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🏷️ Select Tags for Search")

            # Create columns for tag buttons
            top_tags = list(tag_freq.most_common(30))

            # Initialize session state for selected tags
            if "selected_wordcloud_tags" not in st.session_state:
                st.session_state.selected_wordcloud_tags = []

            # Tag selection interface
            cols = st.columns(5)
            for i, (tag, freq) in enumerate(top_tags):
                with cols[i % 5]:
                    button_type = (
                        "primary"
                        if tag in st.session_state.selected_wordcloud_tags
                        else "secondary"
                    )
                    if st.button(f"{tag} ({freq})", key=f"tag_btn_{i}", type=button_type):
                        if tag in st.session_state.selected_wordcloud_tags:
                            st.session_state.selected_wordcloud_tags.remove(tag)
                        else:
                            st.session_state.selected_wordcloud_tags.append(tag)
                        st.rerun()

            # Show selected tags and search results
            if st.session_state.selected_wordcloud_tags:
                st.write("**Selected Tags:**", ", ".join(st.session_state.selected_wordcloud_tags))

                # Clear selection button
                if st.button("🗑️ Clear Selection"):
                    st.session_state.selected_wordcloud_tags = []
                    st.rerun()

                # Search by selected tags
                with st.spinner("Searching by tags..."):
                    tag_results = searcher.search_by_tags(
                        st.session_state.selected_wordcloud_tags, max_results
                    )

                if tag_results:
                    st.success(f"Found {len(tag_results)} documents with " "selected tags")

                    for i, result in enumerate(tag_results, 1):
                        with st.expander(
                            f"📄 {result['filename']} " f"(Score: {result['score']:.3f})"
                        ):
                            col_a, col_b = st.columns([3, 1])

                            with col_a:
                                st.markdown(f"**📁 File:** {result['filename']}")
                                st.markdown(
                                    f"**🏢 Category:** "
                                    f"{result.get('business_category', 'Unknown')}"
                                )
                                st.markdown(
                                    f"**📋 Type:** " f"{result.get('document_type', 'Unknown')}"
                                )
                                st.markdown(
                                    f"**✅ Matching Tags:** "
                                    f"{', '.join(result.get('matching_tags', []))}"
                                )

                                if result.get("tags"):
                                    all_tags_display = ", ".join(result["tags"][:15])
                                    st.markdown(f"**🏷️ All Tags:** " f"{all_tags_display}")

                            with col_b:
                                st.markdown(f"**📊 Tag Match:** " f"{result['score']:.1%}")
                                st.markdown(f"**📝 Words:** " f"{result.get('word_count', 0):,}")
                                st.markdown(
                                    f"**📖 Readability:** " f"{result.get('readability', 0)}"
                                )
                else:
                    st.warning("No documents found with the selected tags")

        with col2:
            st.subheader("📊 Tag Statistics")

            # Show tag frequency chart
            top_10_tags = pd.DataFrame(
                {
                    "Tag": [tag for tag, _ in tag_freq.most_common(10)],
                    "Frequency": [freq for _, freq in tag_freq.most_common(10)],
                }
            )

            fig = px.bar(
                top_10_tags,
                x="Frequency",
                y="Tag",
                orientation="h",
                title="Top 10 Most Frequent Tags",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Tag coverage statistics
            st.markdown("**Tag Coverage:**")
            st.markdown(f"• Total unique tags: {len(tag_freq)}")
            st.markdown(
                f"• Documents with tags: " f"{len([d for d in documents if d.get('tags')])}"
            )
            avg_tags = len([tag for doc in documents for tag in doc.get("tags", [])]) / len(
                documents
            )
            st.markdown(f"• Avg tags per document: {avg_tags:.1f}")

    else:
        st.info(
            "No tags found in the current document set. Tags are "
            "generated automatically when documents are loaded."
        )


def render_analytics_tab(documents: List[Dict[str, Any]]) -> None:
    """Render the analytics tab"""
    st.header("📈 Document Analytics")

    if not documents:
        st.info("No documents to analyze")
        return

    # Display analytics charts
    display_analytics_charts(documents)

    # Display entities
    display_entities(documents)


def render_summary_report_tab(documents: List[Dict[str, Any]], search_method: str) -> None:
    """Render the summary report tab"""
    st.header("📄 Summary Report Generator")
    st.markdown("Create a customized, downloadable summary of your document " "analysis!")

    # Report configuration
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔧 Report Configuration")

        # Report options
        include_details = st.checkbox("Include detailed document information", value=True)
        include_analytics = st.checkbox("Include analytics and statistics", value=True)
        include_tag_cloud = st.checkbox("Include tag analysis", value=True)

        # Search query for report context
        report_query = st.text_input(
            "Search Query (for report context)",
            placeholder=("Optional: Enter the search query this report relates to"),
        )

        # Report format
        report_format = st.selectbox(
            "Report Format", REPORT_FORMATS, help="Choose the format for your downloadable report"
        )

    with col2:
        st.subheader("📊 Report Preview")
        st.metric("Documents in Report", len(documents))
        st.metric("Total Words", f"{sum(doc.get('word_count', 0) for doc in documents):,}")
        st.metric(
            "Categories", len(set([doc.get("business_category", "Unknown") for doc in documents]))
        )
        st.metric(
            "Unique Tags", len(set([tag for doc in documents for tag in doc.get("tags", [])]))
        )

    # Generate report
    if st.button("📊 Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report_data = ReportGenerator.create_summary_report(
                documents=documents,
                search_query=report_query,
                search_method=search_method,
                include_details=include_details,
                include_analytics=include_analytics,
                include_tag_cloud=include_tag_cloud,
            )

            if report_data:
                # Create downloadable content
                downloadable_content = create_downloadable_summary(report_data, report_format)

                if downloadable_content:
                    # Show preview
                    st.subheader("📋 Report Preview")
                    if report_format == "json":
                        st.json(report_data["summary_statistics"])
                    elif report_format == "text":
                        st.text_area("Report Preview", downloadable_content[:2000], height=300)
                    elif report_format == "csv" and "document_details" in report_data:
                        st.dataframe(pd.DataFrame(report_data["document_details"]).head(10))

                    # Download button
                    filename = generate_filename_with_timestamp("document_summary", report_format)

                    st.download_button(
                        label=f"📥 Download {report_format.upper()} Report",
                        data=downloadable_content,
                        file_name=filename,
                        mime=MIME_TYPES[report_format],
                    )

                    st.success(
                        f"Report generated successfully! Click the button "
                        f"above to download your {report_format.upper()} "
                        "report."
                    )
                else:
                    st.error(
                        "Failed to generate downloadable content. Please " "try a different format."
                    )
            else:
                st.error("Failed to generate report. Please ensure documents " "are loaded.")

    # Report template information
    with st.expander("ℹ️ Report Format Information"):
        st.markdown(
            """
        **JSON Format**: Complete structured data including all metadata,
        statistics, and document details. Best for programmatic use or
        importing into other tools.

        **Text Format**: Human-readable summary with key statistics and
        insights. Perfect for sharing with stakeholders or including in
        presentations.

        **CSV Format**: Tabular data of document details that can be
        opened in Excel or other spreadsheet applications. Ideal for
        further data analysis.
        """
        )
