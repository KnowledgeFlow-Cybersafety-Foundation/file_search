"""
Application constants and configuration
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Document Search Dashboard",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Search methods
SEARCH_METHODS = ["keyword", "similarity", "combined", "tags"]

# Report formats
REPORT_FORMATS = ["json", "text", "csv"]

# MIME types for downloads
MIME_TYPES = {
    'json': 'application/json',
    'text': 'text/plain',
    'csv': 'text/csv'
}

# TF-IDF configuration
TFIDF_CONFIG = {
    'max_features': 1000,
    'stop_words': 'english',
    'ngram_range': (1, 2)
}

# Word cloud configuration
WORDCLOUD_CONFIG = {
    'width': 800,
    'height': 400,
    'background_color': 'white',
    'max_words': 100,
    'colormap': 'viridis',
    'relative_scaling': 0.5,
    'random_state': 42
}

# UI configuration
DEFAULT_FOLDER_PATH = "./documents"
MAX_RESULTS_RANGE = (1, 50)
DEFAULT_MAX_RESULTS = 10
READABILITY_RANGE = (0, 100)
DEFAULT_READABILITY_RANGE = (0, 100)
CONTEXT_LENGTH = 200
MAX_CONTEXTS = 3
MAX_TAGS_DISPLAY = 10
MAX_TOPICS_DISPLAY = 3
CONTENT_PREVIEW_LENGTH = 2000
