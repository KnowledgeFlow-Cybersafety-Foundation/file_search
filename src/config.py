from pathlib import Path

# --------- Configuration ---------
PAGE_TITLE = "Document Word Cloud Search"
DOCUMENTS_DIR = Path(__file__).parent / "documents"
TOP_N_WORDS = 50
MIN_WORD_LENGTH = 3

# Extra stopwords that are often useful beyond the default WordCloud set
DEFAULT_EXTRA_STOPWORDS = {"the", "and", "for", "with", "that", "this"}
SIDEBAR_ADD_FILLER_DEFAULT = True
