"""
Reusable UI components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from typing import List, Dict, Any
import re

import sys
sys.path.append('/Users/colinwork/Documents/GitHub/docker_file_search/src')

from utils.helpers import highlight_query_terms, format_tags_display


def display_document_metrics(documents: List[Dict[str, Any]]) -> None:
    """Display document statistics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = len(documents)
    total_words = sum(doc.get('word_count', 0) for doc in documents)
    total_size = sum(doc.get('size_kb', 0) for doc in documents)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    with col1:
        st.metric("📁 Total Documents", total_docs)
    with col2:
        st.metric("📝 Total Words", f"{total_words:,}")
    with col3:
        st.metric("💾 Total Size", f"{total_size:.1f} KB")
    with col4:
        st.metric("📊 Avg Words/Doc", f"{avg_words:,}")


def display_search_result(
    result: Dict[str, Any],
    index: int,
    search_query: str,
    search_method: str
) -> None:
    """Display a single search result"""
    score_display = f"Score: {result['score']:.3f}"
    
    with st.expander(f"📄 {result['filename']} ({score_display})"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**📁 File:** {result['filename']}")
            st.markdown(
                f"**🏢 Category:** "
                f"{result.get('business_category', 'Unknown')}"
            )
            st.markdown(
                f"**📋 Type:** {result.get('document_type', 'Unknown')}"
            )
            st.markdown(f"**📝 Words:** {result['word_count']:,}")
            st.markdown(
                f"**📖 Readability:** {result.get('readability', 0)}"
            )
            st.markdown(
                f"**⚡ Urgency:** {result.get('urgency_level', 'Unknown')}"
            )
            
            # Show tags
            if result.get('tags'):
                tags_display = format_tags_display(result['tags'])
                st.markdown(f"**🏷️ Tags:** {tags_display}")
            
            # Show matching tags for tag search
            if search_method == "tags" and 'matching_tags' in result:
                matching_display = ", ".join(result['matching_tags'])
                st.markdown(f"**✅ Matching Tags:** {matching_display}")
            
            # Show sentiment
            if result.get('sentiment'):
                sentiment = result['sentiment']
                st.markdown(
                    f"**😊 Sentiment:** {sentiment['label']} "
                    f"({sentiment['polarity']:.2f})"
                )
            
            # Show context if available
            if 'context' in result and result['context']:
                st.markdown("**🎯 Context:**")
                for ctx in result['context']:
                    highlighted = ctx
                    for word in search_query.split():
                        highlighted = re.sub(
                            f'({re.escape(word)})',
                            r'**\1**',
                            highlighted,
                            flags=re.IGNORECASE
                        )
                    st.markdown(f"*{highlighted}*")
        
        with col2:
            st.markdown(f"**📊 Match Score:** {result['score']:.3f}")
            if 'matches' in result:
                matches_display = ", ".join(result['matches'])
                st.markdown(f"**🔍 Matched Terms:** {matches_display}")
            
            # Key topics
            if result.get('key_topics'):
                st.markdown("**🎯 Key Topics:**")
                for topic in result['key_topics'][:3]:
                    st.markdown(
                        f"• {topic['topic']} ({topic['frequency']})"
                    )
            
            # Full content preview
            if st.button(
                f"👁️ Preview Full Content",
                key=f"preview_{index}"
            ):
                content_preview = result['content'][:2000]
                if len(result['content']) > 2000:
                    content_preview += "..."
                st.text_area(
                    "Full Document Content",
                    content_preview,
                    height=300,
                    key=f"content_{index}"
                )


def display_category_charts(documents: List[Dict[str, Any]]) -> None:
    """Display category and type distribution charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        if documents:
            categories = [
                doc.get('business_category', 'General')
                for doc in documents
            ]
            category_counts = pd.DataFrame({
                'Category': list(Counter(categories).keys()),
                'Count': list(Counter(categories).values())
            })
            fig = px.pie(
                category_counts,
                values='Count',
                names='Category',
                title='Business Categories'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if documents:
            doc_types = [
                doc.get('document_type', 'Unknown') for doc in documents
            ]
            type_counts = pd.DataFrame({
                'Type': list(Counter(doc_types).keys()),
                'Count': list(Counter(doc_types).values())
            })
            fig = px.bar(
                type_counts,
                x='Type',
                y='Count',
                title='Document Types'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def display_tags_chart(documents: List[Dict[str, Any]]) -> None:
    """Display tags frequency chart"""
    if documents:
        all_tags = [
            tag for doc in documents for tag in doc.get('tags', [])
        ]
        if all_tags:
            st.subheader("🏷️ Top Tags")
            tag_counts = Counter(all_tags)
            top_tags = pd.DataFrame({
                'Tag': list(tag_counts.keys())[:20],
                'Frequency': list(tag_counts.values())[:20]
            })
            fig = px.bar(
                top_tags,
                x='Frequency',
                y='Tag',
                orientation='h',
                title='Most Common Tags'
            )
            st.plotly_chart(fig, use_container_width=True)


def display_analytics_charts(documents: List[Dict[str, Any]]) -> None:
    """Display analytics charts"""
    if not documents:
        return
    
    # Readability analysis
    col1, col2 = st.columns(2)
    with col1:
        readability_scores = [
            doc.get('readability', 0) for doc in documents
        ]
        if readability_scores:
            fig = px.histogram(
                x=readability_scores,
                nbins=20,
                title='Readability Score Distribution',
                labels={'x': 'Readability Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment analysis
        sentiments = [
            doc.get('sentiment', {}).get('label', 'Unknown')
            for doc in documents
        ]
        sentiment_counts = pd.DataFrame({
            'Sentiment': list(Counter(sentiments).keys()),
            'Count': list(Counter(sentiments).values())
        })
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            title='Sentiment Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Document complexity vs readability
    complexity_data = []
    for doc in documents:
        complexity_data.append({
            'Filename': doc.get('filename', ''),
            'Readability': doc.get('readability', 0),
            'Complexity': doc.get('complexity_score', 0),
            'Word Count': doc.get('word_count', 0),
            'Category': doc.get('business_category', 'Unknown')
        })
    
    if complexity_data:
        df_complexity = pd.DataFrame(complexity_data)
        fig = px.scatter(
            df_complexity,
            x='Readability',
            y='Complexity',
            size='Word Count',
            color='Category',
            hover_name='Filename',
            title='Document Complexity vs Readability',
            labels={
                'Readability': 'Readability Score (higher = easier)',
                'Complexity': 'Complexity Score'
            }
        )
        st.plotly_chart(fig, use_container_width=True)


def display_entities(documents: List[Dict[str, Any]]) -> None:
    """Display extracted entities across documents"""
    all_entities = {}
    for doc in documents:
        entities = doc.get('entities', {})
        for entity_type, entity_list in entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = []
            all_entities[entity_type].extend(entity_list)
    
    if all_entities:
        st.subheader("🔍 Extracted Entities")
        entity_cols = st.columns(len(all_entities))
        for i, (entity_type, entities) in enumerate(all_entities.items()):
            with entity_cols[i % len(entity_cols)]:
                st.write(f"**{entity_type.title()}:**")
                unique_entities = list(set(entities))[:5]
                for entity in unique_entities:
                    st.write(f"• {entity}")
