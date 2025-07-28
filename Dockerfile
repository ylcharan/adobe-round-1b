FROM python:3.10-slim
WORKDIR /app

# System libs for OCR + pixmaps (remove if you never OCR or rasterize)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx libgl1 tesseract-ocr libtesseract-dev wget && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App and model
COPY process_collections.py ./
RUN mkdir -p /app/model \
    && wget -O /app/model/model.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf?download=true

# Default entrypoint (adjust flags to match your script)
ENTRYPOINT ["python3","process_collections.py"]
CMD ["--input","/app/input/input.json", "--pdf_folder","/app/input/PDFs", "--output","/app/output/output.json", "--model_path","/app/model/model.gguf"]