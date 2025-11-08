"""
Report generation functionality
"""

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class ReportGenerator:
    """Handle report generation for document analysis"""

    @staticmethod
    def create_summary_report(
        documents: List[Dict[str, Any]],
        search_query: str = "",
        search_method: str = "",
        include_details: bool = True,
        include_analytics: bool = True,
        include_tag_cloud: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Create a comprehensive summary report"""
        if not documents:
            return None

        report_data = {
            "metadata": {
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(documents),
                "search_query": search_query,
                "search_method": search_method,
            },
            "summary_statistics": ReportGenerator._generate_summary_stats(documents),
        }

        if include_details:
            report_data["document_details"] = ReportGenerator._generate_document_details(documents)

        if include_analytics:
            report_data["analytics"] = ReportGenerator._generate_analytics(documents)

        return report_data

    @staticmethod
    def _generate_summary_stats(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_word_count": sum(doc.get("word_count", 0) for doc in documents),
            "average_readability": round(
                np.mean([doc.get("readability", 0) for doc in documents]), 1
            ),
            "total_size_kb": round(sum(doc.get("size_kb", 0) for doc in documents), 2),
            "document_categories": dict(
                Counter([doc.get("business_category", "Unknown") for doc in documents])
            ),
            "document_types": dict(
                Counter([doc.get("document_type", "Unknown") for doc in documents])
            ),
            "urgency_levels": dict(
                Counter([doc.get("urgency_level", "Unknown") for doc in documents])
            ),
            "sentiment_distribution": dict(
                Counter([doc.get("sentiment", {}).get("label", "Unknown") for doc in documents])
            ),
        }

    @staticmethod
    def _generate_document_details(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed document information"""
        document_details = []
        for doc in documents:
            doc_summary = {
                "filename": doc.get("filename", ""),
                "business_category": doc.get("business_category", ""),
                "document_type": doc.get("document_type", ""),
                "word_count": doc.get("word_count", 0),
                "readability": doc.get("readability", 0),
                "urgency_level": doc.get("urgency_level", ""),
                "sentiment": doc.get("sentiment", {}),
                "tags": doc.get("tags", [])[:10],  # Top 10 tags
                "key_topics": doc.get("key_topics", [])[:5],  # Top 5 topics
                "entities": doc.get("entities", {}),
            }
            document_details.append(doc_summary)
        return document_details

    @staticmethod
    def _generate_analytics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analytics section"""
        all_tags = [tag for doc in documents for tag in doc.get("tags", [])]

        readability_scores = [doc.get("readability", 0) for doc in documents]
        complexity_scores = [doc.get("complexity_score", 0) for doc in documents]

        return {
            "top_tags": dict(Counter(all_tags).most_common(20)),
            "readability_stats": {
                "min": min(readability_scores) if readability_scores else 0,
                "max": max(readability_scores) if readability_scores else 0,
                "median": float(np.median(readability_scores)) if readability_scores else 0,
            },
            "complexity_stats": {
                "min": min(complexity_scores) if complexity_scores else 0,
                "max": max(complexity_scores) if complexity_scores else 0,
                "average": round(np.mean(complexity_scores), 1) if complexity_scores else 0,
            },
        }
