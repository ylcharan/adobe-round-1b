# Installation Guide for Challenge 1b

## Requirements

### Minimal Requirements (for current script)

The current `process_collections.py` script only needs:

```bash
pip install PyMuPDF==1.23.14
```

### Full Requirements (for all features)

For the complete feature set including AI capabilities:

```bash
pip install -r requirements.txt
```

## Built-in Python Modules Used

The script uses these built-in Python modules (no installation needed):

- `os` - File system operations
- `json` - JSON file handling
- `datetime` - Timestamp generation

## Installation Steps

### Option 1: Minimal Installation

```bash
cd Challenge_1b
pip install -r requirements_minimal.txt
python3 process_collections.py
```

### Option 2: Full Installation

```bash
cd Challenge_1b
pip install -r requirements.txt
python3 process_collections.py
```

### Option 3: Manual Installation

```bash
pip install PyMuPDF
python3 process_collections.py
```

## Dependency Details

### PyMuPDF (fitz)

- **Purpose**: PDF text extraction and processing
- **Version**: 1.23.14
- **Used for**: Reading PDF files, extracting text with page numbers
- **Alternative**: pdfplumber, pypdf2

### Optional Dependencies

#### ctransformers

- **Purpose**: AI model support for content relevance scoring
- **Note**: Currently not used in the simplified version
- **Used for**: Loading and running LLaMA models

#### nltk/spacy

- **Purpose**: Advanced text processing and NLP
- **Used for**: Better text chunking and keyword extraction

#### tqdm

- **Purpose**: Progress bars for better user experience
- **Used for**: Showing processing progress

## System Requirements

- Python 3.7+
- macOS/Linux/Windows
- At least 500MB free disk space
- 2GB RAM (4GB recommended if using AI features)

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError: No module named 'fitz'**

   ```bash
   pip install PyMuPDF
   ```

2. **Permission errors**

   ```bash
   pip install --user PyMuPDF
   ```

3. **Python version issues**
   - Make sure you're using Python 3.7+
   - Use `python3` instead of `python`

## File Structure

```
Challenge_1b/
├── inputs/
│   ├── input.json
│   └── PDFs/
│       └── (PDF files)
├── outputs/
│   └── output.json (generated)
├── process_collections.py
├── requirements.txt
├── requirements_minimal.txt
└── INSTALL.md (this file)
```
