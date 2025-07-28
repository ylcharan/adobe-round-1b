# Docker Setup for Challenge 1b - PDF Processing

This directory contains a Dockerized solution for processing PDF documents using TinyLlama model.

## Quick Start

### Option 1: Using the Run Script (Recommended)

```bash
# Make sure you're in the Challenge_1b directory
cd Challenge_1b

# Run the automated script
./run_docker.sh
```

### Option 2: Manual Docker Commands

```bash
# Build the Docker image
docker build -t challenge-1b-pdf-processor .

# Run the container
docker run --rm \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  challenge-1b-pdf-processor
```

## Directory Structure

```
Challenge_1b/
├── Dockerfile              # Docker build configuration
├── .dockerignore          # Files to ignore during build
├── run_docker.sh          # Automated run script
├── process_collections.py # Main processing script
├── requirements.txt       # Python dependencies
├── input/                 # PDF input files
│   ├── input.json        # Processing configuration
│   └── PDFs/             # PDF documents to process
├── output/               # Generated JSON results
└── model/                # TinyLlama model files
```

## What the Docker Container Does

1. **Downloads TinyLlama Model**: Automatically downloads `tinyllama-1.1b-chat-v1.0.Q6_K.gguf`
2. **Installs Dependencies**: Sets up Python packages and system dependencies
3. **Processes PDFs**: Runs the collection processing on input PDFs
4. **Generates Output**: Creates structured JSON output with extracted information

## Input Requirements

Make sure your `input/` directory contains:

- `input.json` - Configuration file with task description
- `PDFs/` folder with PDF documents to process

## Output

The container generates:

- `output/output.json` - Structured analysis results matching the required format

## Container Features

- **Optimized Build**: Multi-stage build with minimal final image size
- **OCR Support**: Includes Tesseract for image-based PDFs
- **Model Caching**: Pre-downloads TinyLlama model during build
- **Volume Mounting**: Easy input/output file management
- **Error Handling**: Robust processing with fallback mechanisms

## Troubleshooting

### Model Download Issues

If the model download fails during build:

```bash
# Pre-download the model manually
mkdir -p model
cd model
wget -O tinyllama-1.1b-chat-v1.0.Q6_K.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf
cd ..
```

### Permission Issues

```bash
# Fix permissions if needed
sudo chmod -R 755 input/ output/
```

### Memory Issues

If the container runs out of memory:

```bash
# Run with memory limit
docker run --rm -m 4g \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  challenge-1b-pdf-processor
```

## Environment Variables

- `PYTHONUNBUFFERED=1` - Ensures immediate output logging
- `TESSDATA_PREFIX` - Sets Tesseract OCR data path

## Image Size Optimization

The Dockerfile is optimized to:

- Use Python 3.10 slim base image
- Clean up package managers after installation
- Use multi-layer caching for faster rebuilds
- Minimize final image size while including all dependencies
