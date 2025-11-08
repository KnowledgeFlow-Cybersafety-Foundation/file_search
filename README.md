# Document Search Dashboard

A Streamlit-based application for searching through Word documents with an intuitive web interface.

## New Minimal Word Cloud Search Mode

This repository now includes a simplified Streamlit app (no sidebar) that:

- Loads all `.docx` files from `src/documents/`
- Generates a word cloud of the most frequent words
- Lets you click on top words or manually type a term to search documents
- Shows contextual snippets for each matching document

The minimal interface lives in `src/app.py` (default run target).

## Features

- 📄 **Document Loading**: Automatically load and index all Word documents from a folder
- 🔍 **Multi-Search Methods**:
  - Keyword search for exact matches
  - Similarity search using TF-IDF vectors
  - Combined search for best results
- 📊 **Rich Analytics**: View document statistics and visualizations
- 🎯 **Context Highlighting**: See relevant snippets with search terms highlighted
- 🚀 **Fast Performance**: Efficient indexing and caching for quick searches

## Quick Start

### Using Docker (Recommended)

1. Clone this repository:
```bash
git clone <repository-url>
cd docker_file_search
```

2. Place your Word documents in the `documents/` folder:
```bash
mkdir -p documents
# Copy your .docx files to the documents folder
```

3. Build and run with Docker (word cloud search app):
```bash
docker build -t doc-search .
docker run -p 8501:8501 -v $(pwd)/src/documents:/app/src/documents doc-search
```

4. Open your browser to `http://localhost:8501`

### Local Development

1. Install Python 3.11+ and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app (word cloud search):
```bash
streamlit run src/app.py
```

3. Open your browser to `http://localhost:8501`

## Usage (Word Cloud App)

1. The app automatically loads `.docx` files from `src/documents/` on start.
2. A word cloud is rendered from aggregated text.
3. Click any word button in the "Top Terms" section or type a term into the search box.
4. Matching documents appear with expandable contextual snippets (search term highlighted).
5. Add / remove documents then refresh the page to update the cloud.

## Project Structure

```
docker_file_search/
├── src/
│   ├── app.py          # Minimal word cloud search application (no sidebar)
│   ├── utils.py        # Document processing utilities
│   └── config.py       # Configuration settings
├── documents/          # Place your Word documents here
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
└── README.md          # This file
```

## Search Methods

- **Keyword**: Simple text matching with word counting
- **Similarity**: TF-IDF based semantic similarity
- **Combined**: Merges both methods for comprehensive results

## Requirements

- Python 3.11+
- Streamlit 1.31.0+
- python-docx for Word document processing
- scikit-learn for similarity search
- pandas and plotly for data visualization

## Docker Support

The application includes a Dockerfile for easy deployment:

- Based on Python 3.11 slim image
- Exposes port 8501 for Streamlit
- Optimized for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
