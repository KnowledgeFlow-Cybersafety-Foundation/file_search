"""Configuration settings for the Document Search application"""

# Search settings
DEFAULT_SEARCH_METHOD = "combined"
MAX_RESULTS_DEFAULT = 10
CONTEXT_LENGTH = 200

# TF-IDF settings
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = 'english'

# File settings
SUPPORTED_EXTENSIONS = ['.docx']
DEFAULT_DOCUMENTS_FOLDER = './documents'

# UI settings
PAGE_TITLE = "Document Search Dashboard"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Performance settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_FILE_SIZE_MB = 10