import os
import sys
import re
import io
import base64
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

# ────────────────────────────────
# 0. ENV PATCHES
# ────────────────────────────────
os.environ.setdefault("STREAMLIT_FILE_WATCHER_TYPE", "none")
import types
sys.modules.setdefault("torch.classes", types.ModuleType("torch.classes"))

# ────────────────────────────────
# 1. CONFIG & INIT
# ────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

from itertools import islice  # for chunking pages

def clean_text(text: str) -> str:
    """Trim blank lines and strip URLs."""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln and not re.match(r"^https?://\S+$", ln))

def llama_ocr_page(page: fitz.Page) -> Image.Image:
    """Render a PDF page to a PIL image at 2× resolution."""
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def encode_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def extract_input(file):
    """Return a list of PIL images for PDF pages or uploaded image."""
    if file is None:
        return None
    ctype = file.type
    data = file.read()
    if "pdf" in ctype:
        doc = fitz.open(stream=data, filetype="pdf")
        return [llama_ocr_page(page) for page in doc]
    elif "image" in ctype:
        return [Image.open(io.BytesIO(data))]
    else:
        st.error("Unsupported file type.")
        return None

# ────────────────────────────────
# 2. STREAMLIT UI
# ────────────────────────────────
st.set_page_config("Exam Evaluator", layout="wide")
st.title("🧾 AI Exam Evaluator — Vision-Only (No Reference)")

with st.sidebar:
    st.header("📥 Upload Exam Paper")
    paper_file = st.file_uploader(
        "Upload the exam paper (with questions, student answers, and marks) as PDF or Image",
        type=["pdf", "png", "jpg", "jpeg"],
    )
    temperature = st.slider("🎛️ LLM temperature", 0.0, 1.0, 0.5, 0.05)
    run_btn = st.button("🚀 Evaluate")

# ────────────────────────────────
# 3. INPUT EXTRACTION & PREVIEW
# ────────────────────────────────
pages = extract_input(paper_file)

if pages:
    st.subheader("📄 Uploaded Exam Paper Preview")
    cols = st.columns(len(pages))
    for col, img in zip(cols, pages):
        col.image(img, use_container_width=True)

# ────────────────────────────────
# 4. CHUNKING FUNCTION
# ────────────────────────────────
def chunk_pages(lst, size=5):
    """Yield successive size-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

# ────────────────────────────────
# 5. EVALUATION LOGIC
# ────────────────────────────────
def evaluate_pages_with_vision(pages, temp):
    """
    Sends pages in batches (≤5 images) to the Vision LLM.
    Aggregates partial reports into one final report.
    """
    reports = []
    system_prompt = (
        "You are an expert exam evaluator. The provided document images contain:\n"
        "  • Questions (with allocated marks clearly indicated)\n"
        "  • Student answers written beneath each question\n\n"
        "Your tasks:\n"
        "  1. Identify each question, its marking allocation, and the student's answer.\n"
        "  2. Grade each answer against its allocation, giving a concise rationale. Do not grade if the answers are not provided.\n"
        "  3. Provide a section-wise breakdown and a final total score out of the sum of allocated marks.\n"
        "  4. Present the result in a clear report: question number, marks awarded/allocated, feedback.\n"
    )

    for chunk in chunk_pages(pages, size=5):
        content = []
        for idx, img in enumerate(chunk, start=1):
            content.append({"type": "text",      "text": f"--- Page {idx} ---"})
            content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(img)}})

        resp = client.chat.completions.create(
            model=SCOUT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content},
            ],
            temperature=temp,
            max_tokens=2048,
        )
        reports.append(resp.choices[0].message.content.strip())

    return "\n\n".join(reports)

# ────────────────────────────────
# 6. RUN EVALUATION
# ────────────────────────────────
if run_btn:
    if not pages:
        st.error("Please upload an exam paper first.")
        st.stop()

    with st.spinner("🤖 Evaluating exam..."):
        try:
            report = evaluate_pages_with_vision(pages, temperature)
            st.subheader("📊 Evaluation Report")
            st.code(report, language="markdown")
            st.download_button(
                "📥 Download Report",
                report,
                file_name="evaluation_report.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
