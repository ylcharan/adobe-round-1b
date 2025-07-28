#!/usr/bin/env python3
"""
PDF → (native text ⟶ or ⟶ image OCR) → chunk → keyword/LLM rank → summary

• Native selectable text is used when present.
• Pages that are image‑only (or when `ocr_every_page=True`) are rasterised and
  passed through Tesseract.
• Optional ctransformers‑based Llama model can rerank chunks and refine output.
"""

import os, argparse
import json
from datetime import datetime
import fitz                                 # PyMuPDF
from PIL import Image
import pytesseract

# ────────────────────────────────────────────────────────────────────────────────
# Optional local Llama model via ctransformers
# ────────────────────────────────────────────────────────────────────────────────
try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    CTRANSFORMERS_AVAILABLE = False


parser = argparse.ArgumentParser()
parser.add_argument("--input",      required=True)
parser.add_argument("--pdf_folder", required=True)
parser.add_argument("--output",     required=True)
parser.add_argument("--model_path",
                    default=os.path.join(os.path.dirname(__file__),
                                         "model","model.gguf"),
                    help="Path to .gguf model file")
args = parser.parse_args()

# def load_model():
#     """Searches ./model/ for a *.gguf file and loads it with ctransformers."""
#     cwd = os.getcwd()
#     print(f"🗺️  CWD: {cwd}")
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     model_dir = os.path.join(script_dir, "model")
#     if not os.path.isdir(model_dir):
#         print(f"❌  model/ directory not found: {model_dir}")
#         return None

#     if not CTRANSFORMERS_AVAILABLE:
#         print("❌  ctransformers not installed — pip install ctransformers")
#         return None

#     for name in os.listdir(model_dir):
#         if name.lower().endswith(".gguf"):
#             path = os.path.join(model_dir, name)
#             print(f"🔍 Loading {path}")
#             try:
#                 if CTRANSFORMERS_AVAILABLE and os.path.exists(model_path):
#                     model = AutoModelForCausalLM.from_pretrained(
#                         path,
#                         model_type="llama",
#                         gpu_layers=0,
#                         context_length=1024,
#                         threads=4,
#                         batch_size=1
#                     )
#                     # smoke‑test
#                     try:
#                         model.generate("Hello", n_predict=5)
#                     except TypeError:
#                         model("Hello", max_new_tokens=5)
#                     print("✅  Llama model loaded")
#                     return model
#             except Exception as e:
#                 print(f"❌  Failed to load {name}: {e}")
#     print("❌  No usable .gguf model found")
#     return None

def load_model(model_path):
    if not CTRANSFORMERS_AVAILABLE:
        print("❌  ctransformers not installed")
        return None

    if not os.path.isfile(model_path):
        print(f"❌  No model found at {model_path}")
        return None

    print(f"🔍  Loading Llama model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        gpu_layers=0,
        context_length=1024,
        threads=4,
        batch_size=1
    )
    # smoke‑test…
    try:
        model.generate("Hello", n_predict=5)
    except TypeError:
        model("Hello", max_new_tokens=5)
    print("✅  Llama model loaded")
    return model



# ────────────────────────────────────────────────────────────────────────────────
# PDF text extraction helpers
# ────────────────────────────────────────────────────────────────────────────────
def extract_page_text(page, force_ocr=False, dpi=200):
    """
    Returns text from a single PyMuPDF page.

    • If `force_ocr` is False we first try the fast, native `page.get_text()`.
    • If that is empty OR `force_ocr` is True, we rasterise the page and run
      it through Tesseract.
    """
    txt = page.get_text().strip()
    if txt and not force_ocr:
        return txt

    # OCR branch ───────────────────────────────────────────────────────────────
    zoom = dpi / 72                     # 72 dpi = PDF user‑space resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)


def load_pdf_text_with_pages(pdf_path, ocr_every_page=False, dpi=200):
    """
    Reads a PDF and returns:
        [ { 'page_number': int, 'text': str }, ... ]

    Set `ocr_every_page=True` to force OCR on *every* page.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = extract_page_text(page,
                                 force_ocr=ocr_every_page,
                                 dpi=dpi)
        pages.append({'page_number': i, 'text': text})
    return pages


# ────────────────────────────────────────────────────────────────────────────────
# Chunking + relevance scoring
# ────────────────────────────────────────────────────────────────────────────────
def split_text_with_pages(pages, chunk_size=500, overlap=50):
    """
    Break long page text into overlapping chunks keeping page numbers.
    """
    chunks = []
    for p in pages:
        txt = p['text']
        for i in range(0, len(txt), chunk_size - overlap):
            seg = txt[i:i + chunk_size]
            if seg.strip():
                chunks.append({'text': seg,
                               'page_number': p['page_number']})
    return chunks


def calculate_keyword_score(chunk, task):
    text_l = chunk['text'].lower()
    score = 0
    for kw in set(task.lower().split()):
        score += text_l.count(kw)
    return score


def find_relevant_chunks_enhanced(task, chunks, model=None, top_k=5):
    scored = [{'chunk': c, 'score': calculate_keyword_score(c, task)}
              for c in chunks]
    scored = [x for x in scored if x['score'] > 0]
    scored.sort(key=lambda x: x['score'], reverse=True)
    candidates = scored[:top_k * 2]

    # ── optional LLM re‑ranking ───────────────────────────────────────────────
    if model and candidates:
        print("🤖  Llama re‑ranking top keyword matches")
        print("🔍  Using Llama model for relevance scoring...")
        import re
        rescored = []
        for item in candidates:
            snippet = item['chunk']['text'][:200]
            prompt = (f"Rate relevance 1‑10 for task: {task}\n"
                      f"-----\n{snippet}\n-----\nScore:")
            try:
                try:
                    out = model.generate(prompt, n_predict=3)
                except TypeError:
                    out = model(prompt, max_new_tokens=3)
                nums = re.findall(r"\b(?:10|[1-9])\b", out)
                llm_score = float(nums[0]) if nums else item['score']
            except Exception:
                llm_score = item['score']
            rescored.append({'chunk': item['chunk'], 'score': llm_score})
        rescored.sort(key=lambda x: x['score'], reverse=True)
        return rescored[:top_k]

    return candidates[:top_k]


# ────────────────────────────────────────────────────────────────────────────────
# Refinement / summarisation
# ────────────────────────────────────────────────────────────────────────────────
def refine_text_enhanced(text, task, persona, model=None):
    if model:
        prompt = (f"Act as a {persona}. You are {persona}. Your task is to {task}.Now, perform the task.")
        try:
            try:
                out = model.generate(prompt, n_predict=50)
            except TypeError:
                out = model(prompt, max_new_tokens=50)
            if out and len(out.strip()) > 20:
                return out.strip()
        except Exception:
            pass

    # fallback: first two sentences
    sents = [s.strip() for s in text.replace('\n', ' ').split('.')
             if s.strip()]
    return '. '.join(sents[:2]) + ('.' if sents[:2] else '')


# ────────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────────
def process_collection_enhanced(input_file, output_file, pdf_folder,
                                ocr_every_page=False, dpi=200, model=None):
    print(f"📥  Loading instructions: {input_file}")
    print(f"📂  Source PDFs:         {pdf_folder}")
    print(f"📤  Output JSON:         {output_file}\n")

    model = load_model(args.model_path)

    with open(input_file) as f:
        cfg = json.load(f)

    task    = cfg['job_to_be_done']['task']
    persona = cfg['persona']['role']
    docs    = cfg['documents']

    all_chunks = []
    for d in docs:
        pdf_path = os.path.join(pdf_folder, d['filename'])
        if not os.path.exists(pdf_path):
            print(f"❌  Missing PDF: {pdf_path}")
            continue
        pages = load_pdf_text_with_pages(pdf_path,
                                         ocr_every_page=ocr_every_page,
                                         dpi=dpi)
        chunks = split_text_with_pages(pages)
        for c in chunks:
            c['document'] = d['filename']
        all_chunks.extend(chunks)

    print(f"📊  Extracted {len(all_chunks)} chunks")

    relevant = find_relevant_chunks_enhanced(task, all_chunks,
                                             model=model, top_k=5)

    result = {
        'metadata': {
            'input_documents': [d['filename'] for d in docs],
            'persona': persona,
            'job_to_be_done': task,
            'processing_timestamp': datetime.now().isoformat()
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }

    for rank, item in enumerate(relevant, start=1):
        chk = item['chunk']
        summary = refine_text_enhanced(chk['text'], task, persona, model=model)
        result['extracted_sections'].append({
            'document': chk['document'],
            'section_title': 'Extracted Section',
            'importance_rank': rank,
            'page_number': chk['page_number']
        })
        result['subsection_analysis'].append({
            'document': chk['document'],
            'refined_text': summary,
            'page_number': chk['page_number'],
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print("✅  Done — results saved.\nYou can now use the output JSON for further processing.")
    return result


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = load_model(args.model_path)

    process_collection_enhanced(
        input_file='input/input.json',
        pdf_folder='input/PDFs',
        output_file='output/output.json',
        ocr_every_page=False,  # set True to OCR all pages
        dpi=300,          # increase for finer OCR if needed
        model=model
    )
